#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, Subset

# Make project imports work when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transforms as T
from dataset_scene import build_scene_graph_data
from models import build_model
from utils import image_graph_collate_scene_graph

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Obj:
    pass


def dict_to_obj(x):
    if isinstance(x, dict):
        o = Obj()
        for k, v in x.items():
            setattr(o, k, dict_to_obj(v))
        return o
    if isinstance(x, list):
        return [dict_to_obj(v) for v in x]
    return x


def setup_distributed(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available() and device_arg.startswith("cuda")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for k in ("net", "model", "state_dict"):
            if k in checkpoint_obj and isinstance(checkpoint_obj[k], dict):
                return checkpoint_obj[k]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def maybe_strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def denorm_image_tensor_to_uint8(tensor):
    t = tensor.detach().cpu().float().clone()
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3 or t.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(t.shape)}")
    for i, (m, s) in enumerate(zip(MEAN, STD)):
        t[i] = t[i] * s + m
    img = t.permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def apply_trigger_to_pil(image_pil, patch_size=32, margin=10):
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    img = image_pil.copy()
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    p = int(min(patch_size, h, w))
    if p <= 0:
        return img
    x0 = max(0, min(int(margin), w - p))
    y0 = max(0, h - int(margin) - p)
    x1, y1 = x0 + p, y0 + p
    # spring-green patch (same style as current relationformer trigger)
    arr[y0:y1, x0:x1, 0] = 0
    arr[y0:y1, x0:x1, 1] = 255
    arr[y0:y1, x0:x1, 2] = int(round(0.3 * 255))
    return Image.fromarray(arr, mode="RGB")


def build_single_transform():
    return T.Compose(
        [
            T.RandomResize([800], max_size=800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def save_heatmap_png(hm: torch.Tensor, title: str, out_path: str, cmap: str = "YlOrBr"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4.2, 4.0))
    plt.imshow(hm.detach().cpu(), cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def unpack_relation_cross_attn(out_dict):
    attn_map = out_dict.get("attn_map", None)
    if isinstance(attn_map, (tuple, list)) and len(attn_map) >= 2:
        return attn_map[1]
    raise RuntimeError(
        "Cannot find relation-token cross attention in model output['attn_map']."
    )


@torch.no_grad()
def cross_attn_heatmaps(
    img_attn: torch.Tensor,  # [B,Q,S] or [B,H,Q,S]
    memory_spatial_shapes: torch.Tensor,  # [L,2] or [B,L,2]
    memory_level_start_index: torch.Tensor,  # [L]
    b: int = 0,
    q: int = 0,
    memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B,S]
    per_level_norm: str = "none",  # "none" / "max" / "sum"
    mode: str = "both",  # "per_level" / "fused" / "both"
    apply_softmax: bool = False,
    fuse_to_level0: bool = True,
    eps: float = 1e-12,
):
    # 1) attention vector [S]
    if img_attn.dim() == 3:
        w = img_attn[b, q].detach().float()
    elif img_attn.dim() == 4:
        w = img_attn[b, :, q].mean(0).detach().float()
    else:
        raise ValueError(f"Unexpected img_attn shape: {tuple(img_attn.shape)}")
    S = w.numel()

    # 2) optional key padding mask
    pad = None
    if memory_key_padding_mask is not None:
        pad = memory_key_padding_mask[b].to(torch.bool)

    # 3) spatial shapes
    if memory_spatial_shapes.dim() == 3:
        spatial_shapes = memory_spatial_shapes[b]
    else:
        spatial_shapes = memory_spatial_shapes

    L = spatial_shapes.shape[0]
    start = memory_level_start_index.to(torch.long)
    lengths = torch.empty((L,), device=start.device, dtype=torch.long)
    lengths[:-1] = start[1:] - start[:-1]
    lengths[-1] = S - start[-1]

    # 4) softmax logic (relationformer img_attn is already softmax weights by default)
    if apply_softmax:
        if pad is not None:
            w = w.masked_fill(pad, float("-inf"))
        w = torch.softmax(w, dim=-1)
        if pad is not None:
            w = w.masked_fill(pad, 0.0)
    else:
        if pad is not None:
            w = w.masked_fill(pad, 0.0)

    # 5) per-level raw maps
    per_level_raw: List[torch.Tensor] = []
    for lv in range(L):
        H, W = spatial_shapes[lv].tolist()
        s = int(start[lv].item())
        n = int(lengths[lv].item())
        if H * W != n:
            raise RuntimeError(
                f"Level {lv}: H*W={H*W} vs slice={n} (shape/index mismatch)"
            )
        per_level_raw.append(w[s : s + n].reshape(H, W))

    # 6) per-level display normalization
    per_level_show: List[torch.Tensor] = []
    for hm in per_level_raw:
        if per_level_norm == "none":
            x = hm
        elif per_level_norm == "max":
            x = hm - hm.min()
            x = x / (x.max() + eps)
        elif per_level_norm == "sum":
            x = hm.clamp_min(0.0)
            x = x / (x.sum() + eps)
        else:
            raise ValueError("per_level_norm must be one of: none/max/sum")
        per_level_show.append(x)

    # 7) fused map (upsample to level-0, no mass conservation)
    fused = None
    if mode in ("fused", "both"):
        if fuse_to_level0:
            H0, W0 = spatial_shapes[0].tolist()
        else:
            H0, W0 = spatial_shapes[0].tolist()
        fused = torch.zeros((H0, W0), device=per_level_raw[0].device, dtype=per_level_raw[0].dtype)
        for hm in per_level_raw:
            up = F.interpolate(
                hm[None, None], size=(H0, W0), mode="bilinear", align_corners=False
            )[0, 0]
            fused = fused + up

    if mode == "per_level":
        return per_level_show, None
    if mode == "fused":
        return [], fused
    return per_level_show, fused


class DecoderLayerCapture:
    def __init__(self, model, layer_index_1based: int):
        layers = model.transformer.decoder.layers
        num_layers = len(layers)
        if num_layers <= 0:
            raise RuntimeError("Decoder has no layers.")
        if layer_index_1based < 1 or layer_index_1based > num_layers:
            raise ValueError(
                f"--decoder_layer must be in [1, {num_layers}], got {layer_index_1based}"
            )

        self.layer_index_1based = int(layer_index_1based)
        self.layer_index_0based = int(layer_index_1based - 1)
        self.cache = {}
        self.handles = []

        layer = layers[self.layer_index_0based]
        self.handles.append(layer.register_forward_pre_hook(self._layer_pre_hook))
        self.handles.append(layer.multi_head_cross_attn.register_forward_hook(self._cross_attn_hook))

    def _layer_pre_hook(self, _module, inputs):
        # forward signature:
        # (tgt, query_pos, reference_points, src, src_spatial_shapes,
        #  level_start_index, pos, src_padding_mask)
        if len(inputs) < 8:
            return
        self.cache["spatial_shapes"] = inputs[4].detach()
        self.cache["level_start_index"] = inputs[5].detach()
        self.cache["padding_mask"] = None if inputs[7] is None else inputs[7].detach()

    def _cross_attn_hook(self, _module, _inputs, output):
        # output is (attn_output, attn_weights)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            self.cache["img_attn"] = output[1].detach()

    def get(self):
        required = ["img_attn", "spatial_shapes", "level_start_index"]
        missing = [k for k in required if k not in self.cache]
        if missing:
            raise RuntimeError(
                f"Did not capture required tensors from decoder layer {self.layer_index_1based}: {missing}"
            )
        return (
            self.cache["img_attn"],
            self.cache["spatial_shapes"],
            self.cache["level_start_index"],
            self.cache.get("padding_mask", None),
        )

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@dataclass
class Aggregator:
    stats_size: int
    device: torch.device
    trigger_patch_size: int
    trigger_margin: int
    topk_ratio: float = 0.002
    topk_label: str = "0_2_percent"

    def __post_init__(self):
        self.sum_map = torch.zeros(
            (self.stats_size, self.stats_size), dtype=torch.float32, device=self.device
        )
        self.count = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_mass_sum = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_metric_count = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_box_xyxy_01_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)

    def add(self, fused_map: Optional[torch.Tensor], image_hw: Optional[Tuple[int, int]] = None):
        if fused_map is None:
            return
        x = fused_map.detach().float().clamp_min(0.0)
        s = float(x.sum().item())
        if s <= 0.0:
            return
        x = x / (x.sum() + 1e-12)
        x = F.interpolate(
            x[None, None],
            size=(self.stats_size, self.stats_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        x = x.clamp_min(0.0)
        x = x / (x.sum() + 1e-12)
        self.sum_map += x
        self.count += 1.0
        self._accumulate_trigger_metrics(x, image_hw)

    def _accumulate_trigger_metrics(self, x: torch.Tensor, image_hw: Optional[Tuple[int, int]] = None):
        if image_hw is None:
            return
        h_img, w_img = int(image_hw[0]), int(image_hw[1])
        if h_img <= 0 or w_img <= 0:
            return

        ps = int(max(1, min(self.trigger_patch_size, h_img, w_img)))
        mg = int(max(0, self.trigger_margin))
        x0_px = int(max(0, min(mg, w_img - ps)))
        y0_px = int(max(0, h_img - mg - ps))
        x1_px = int(x0_px + ps)
        y1_px = int(y0_px + ps)

        h_map, w_map = int(x.shape[0]), int(x.shape[1])
        ix0 = int(np.floor((x0_px / float(w_img)) * w_map))
        iy0 = int(np.floor((y0_px / float(h_img)) * h_map))
        ix1 = int(np.ceil((x1_px / float(w_img)) * w_map))
        iy1 = int(np.ceil((y1_px / float(h_img)) * h_map))
        ix0 = max(0, min(ix0, w_map - 1))
        iy0 = max(0, min(iy0, h_map - 1))
        ix1 = max(ix0 + 1, min(ix1, w_map))
        iy1 = max(iy0 + 1, min(iy1, h_map))

        trigger_mass = x[iy0:iy1, ix0:ix1].sum()
        box_xyxy_01 = torch.tensor(
            [
                x0_px / float(w_img),
                y0_px / float(h_img),
                x1_px / float(w_img),
                y1_px / float(h_img),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.trigger_mass_sum += trigger_mass
        self.trigger_metric_count += 1.0
        self.trigger_box_xyxy_01_sum += box_xyxy_01

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.sum_map, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.trigger_mass_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.trigger_metric_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.trigger_box_xyxy_01_sum, op=dist.ReduceOp.SUM)


def compute_stats(mean_map: np.ndarray, agg: Aggregator):
    total = float(mean_map.sum())
    if total <= 0.0:
        return {}
    p = mean_map / total

    flat = p.reshape(-1)
    k_top = max(1, int(np.ceil(agg.topk_ratio * flat.size)))
    topk_mass = float(np.partition(flat, -k_top)[-k_top:].sum())

    trigger_stats = {
        "position": "bottom_left",
        "margin_px": int(agg.trigger_margin),
        "patch_size_px": int(agg.trigger_patch_size),
        "num_images_with_trigger_metrics": int(round(float(agg.trigger_metric_count.item()))),
    }
    if float(agg.trigger_metric_count.item()) > 0:
        c = agg.trigger_metric_count.clamp_min(1.0)
        mean_box = (agg.trigger_box_xyxy_01_sum / c).detach().cpu().numpy()
        trigger_stats.update(
            {
                "trigger_box_xyxy_01_mean": {
                    "x0": float(mean_box[0]),
                    "y0": float(mean_box[1]),
                    "x1": float(mean_box[2]),
                    "y1": float(mean_box[3]),
                },
                "trigger_mass_mean": float((agg.trigger_mass_sum / c).item()),
            }
        )

    return {
        "topk_mass": {
            f"top_{agg.topk_label}": topk_mass,
            f"k_pixels_top_{agg.topk_label}": int(k_top),
        },
        "trigger_region": trigger_stats,
    }


def save_summary(output_dir: str, agg: Aggregator):
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    n = int(round(float(agg.count.item())))
    if n <= 0:
        payload = {"num_images_captured": 0}
        with open(os.path.join(stats_dir, "attention_stats.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    mean_map = (agg.sum_map / agg.count.clamp_min(1.0)).detach().cpu().numpy()
    mean_map = np.clip(mean_map, 0.0, None)
    mean_map = mean_map / (mean_map.sum() + 1e-12)

    np.save(os.path.join(stats_dir, "mean_attention.npy"), mean_map)
    plt.figure(figsize=(5.0, 4.0))
    plt.imshow(mean_map, cmap="YlOrBr", interpolation="nearest")
    plt.title("Mean Relation-Token Cross-Attention (FUSED)")
    plt.axis("off")
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, "mean_attention.png"), dpi=220, bbox_inches="tight")
    plt.close()

    payload = {
        "num_images_captured": n,
        "stats_size": int(agg.stats_size),
        "stats": compute_stats(mean_map, agg=agg),
    }
    with open(os.path.join(stats_dir, "attention_stats.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def build_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    config = dict_to_obj(cfg)
    if not hasattr(config, "BACKDOOR") or config.BACKDOOR is None:
        config.BACKDOOR = Obj()
    config.BACKDOOR.is_bd = bool(args.backdoor_eval)
    config.BACKDOOR.trigger_object_idx = int(args.trigger_object_idx)
    config.BACKDOOR.trigger_predicate_idx = int(args.trigger_predicate_idx)
    return config


def build_model_and_load(config, checkpoint_path: str, device: torch.device):
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(ckpt)
    state = maybe_strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded from: {checkpoint_path}")
    if len(missing) > 0:
        print(f"[ckpt] missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def select_subset_indices(n_total: int, n_pick: int, seed: int):
    n = min(int(n_pick), int(n_total))
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n, replace=False)
    return indices.tolist()


def parse_batch(batch, use_backdoor_images: bool):
    if len(batch) >= 4:
        if use_backdoor_images:
            images, targets = list(batch[2]), list(batch[3])
            branch = "backdoor"
        else:
            images, targets = list(batch[0]), list(batch[1])
            branch = "clean"
    else:
        images, targets = list(batch[0]), list(batch[1])
        branch = "clean"
    return images, targets, branch


def maybe_get_image_id(target, default_idx: int):
    if isinstance(target, dict) and "image_id" in target:
        t = target["image_id"]
        if torch.is_tensor(t):
            return int(t.item())
        return int(t)
    return int(default_idx)


def run_single(args, model, device, capture, agg, rank: int):
    if not args.image:
        raise ValueError("--image is required in --mode single")
    if rank != 0:
        return

    per_image_dir = os.path.join(args.output_dir, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)

    image_pil = Image.open(args.image).convert("RGB")
    if args.apply_trigger_single:
        image_pil = apply_trigger_to_pil(
            image_pil, patch_size=args.trigger_patch_size, margin=args.trigger_margin
        )

    img_tensor, _ = build_single_transform()(image_pil, None)
    images = [img_tensor.to(device)]

    with torch.no_grad():
        _ = model(images)
    img_attn, spatial_shapes, level_start, key_padding_mask = capture.get()

    compute_mode = "both" if args.visualize_mode == "per_level" else args.visualize_mode
    per_levels, fused = cross_attn_heatmaps(
        img_attn=img_attn,
        memory_spatial_shapes=spatial_shapes,
        memory_level_start_index=level_start,
        b=0,
        q=0,
        memory_key_padding_mask=key_padding_mask,
        per_level_norm=args.per_level_norm,
        mode=compute_mode,
        apply_softmax=args.apply_softmax,
    )

    stem = os.path.splitext(os.path.basename(args.image))[0]
    key = f"single_{stem}"
    if args.save_input_image:
        input_png = os.path.join(per_image_dir, f"{key}_input.png")
        Image.fromarray(denorm_image_tensor_to_uint8(images[0])).save(input_png)

    if args.visualize_mode in ("per_level", "both"):
        for lv, hm in enumerate(per_levels):
            out_png = os.path.join(per_image_dir, f"{key}_lv{lv}.png")
            save_heatmap_png(hm, f"Level{lv} (H={hm.shape[0]}, W={hm.shape[1]})", out_png)
    if args.visualize_mode in ("fused", "both") and fused is not None:
        fused_vis = fused.clamp_min(0.0)
        fused_vis = fused_vis / (fused_vis.sum() + 1e-12)
        out_png = os.path.join(per_image_dir, f"{key}_fused_to_lv0.png")
        save_heatmap_png(
            fused_vis, f"FUSED -> Level0 (H={fused.shape[0]}, W={fused.shape[1]})", out_png
        )
    image_hw = (int(images[0].shape[-2]), int(images[0].shape[-1]))
    agg.add(fused, image_hw=image_hw)


def run_dataset(args, config, model, device, capture, agg, rank: int, world_size: int):
    _, val_ds = build_scene_graph_data(
        config=config,
        mode="split",
        debug=bool(args.debug),
        do_bd=bool(args.backdoor_eval),
        trigger_object_idx=int(args.trigger_object_idx),
        trigger_predicate_idx=int(args.trigger_predicate_idx),
    )
    indices = select_subset_indices(len(val_ds), args.num_samples, args.sample_seed)
    subset = Subset(val_ds, indices)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=image_graph_collate_scene_graph,
        pin_memory=True,
    )

    per_image_dir = os.path.join(args.output_dir, "per_image")
    if args.save_individual and rank == 0:
        os.makedirs(per_image_dir, exist_ok=True)

    saved = 0
    global_i = 0
    with torch.no_grad():
        for batch in loader:
            images_cpu, targets, branch = parse_batch(batch, args.use_backdoor_images)
            images = [img.to(device, non_blocking=True) for img in images_cpu]

            _ = model(images)
            img_attn, spatial_shapes, level_start, key_padding_mask = capture.get()

            bs = len(images)
            for b in range(bs):
                compute_mode = "both" if args.visualize_mode == "per_level" else args.visualize_mode
                per_levels, fused = cross_attn_heatmaps(
                    img_attn=img_attn,
                    memory_spatial_shapes=spatial_shapes,
                    memory_level_start_index=level_start,
                    b=b,
                    q=0,
                    memory_key_padding_mask=key_padding_mask,
                    per_level_norm=args.per_level_norm,
                    mode=compute_mode,
                    apply_softmax=args.apply_softmax,
                )
                image_hw = (int(images[b].shape[-2]), int(images[b].shape[-1]))
                agg.add(fused, image_hw=image_hw)

                if not args.save_individual or rank != 0:
                    global_i += 1
                    continue

                img_id = maybe_get_image_id(targets[b], default_idx=global_i)
                key = f"{branch}_id{img_id:06d}_rank{rank}"
                if args.save_input_image:
                    input_png = os.path.join(per_image_dir, f"{key}_input.png")
                    Image.fromarray(denorm_image_tensor_to_uint8(images[b])).save(input_png)

                if args.visualize_mode in ("per_level", "both"):
                    for lv, hm in enumerate(per_levels):
                        out_png = os.path.join(per_image_dir, f"{key}_lv{lv}.png")
                        save_heatmap_png(
                            hm, f"Level{lv} (H={hm.shape[0]}, W={hm.shape[1]})", out_png
                        )
                if args.visualize_mode in ("fused", "both") and fused is not None:
                    fused_vis = fused.clamp_min(0.0)
                    fused_vis = fused_vis / (fused_vis.sum() + 1e-12)
                    out_png = os.path.join(per_image_dir, f"{key}_fused_to_lv0.png")
                    save_heatmap_png(
                        fused_vis,
                        f"FUSED -> Level0 (H={fused.shape[0]}, W={fused.shape[1]})",
                        out_png,
                    )
                saved += 1
                global_i += 1

    if rank == 0:
        print(
            f"[dataset] sampled={len(indices)} branch={'backdoor' if args.use_backdoor_images else 'clean'} "
            f"save_individual={args.save_individual} saved={saved}"
        )


def get_parser():
    p = argparse.ArgumentParser(
        "Standalone RelationFormer relation-token cross-attention visualization"
    )
    p.add_argument("--mode", default="single", choices=["single", "dataset"])
    p.add_argument("--config", required=True, help="relationformer yaml config")
    p.add_argument("--checkpoint", required=True, help="model checkpoint path")
    p.add_argument("--output_dir", default="./relationformer/debug_vis/relation_attention_vis")
    p.add_argument("--device", default="cuda")

    p.add_argument("--image", default="", help="single mode image path")
    p.add_argument("--apply_trigger_single", action="store_true")
    p.add_argument("--trigger_patch_size", type=int, default=32)
    p.add_argument("--trigger_margin", type=int, default=10)

    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--sample_seed", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--debug", action="store_true")

    p.add_argument("--backdoor_eval", action="store_true")
    p.add_argument("--use_backdoor_images", action="store_true")
    p.add_argument("--trigger_object_idx", type=int, default=26)
    p.add_argument("--trigger_predicate_idx", type=int, default=23)

    p.add_argument("--visualize_mode", default="both", choices=["per_level", "fused", "both"])
    p.add_argument(
        "--decoder_layer",
        type=int,
        default=-1,
        help="1-based decoder layer index to visualize; set -1 to use the last layer.",
    )
    p.add_argument("--per_level_norm", default="none", choices=["none", "max", "sum"])
    p.add_argument(
        "--apply_softmax",
        action="store_true",
        help="Apply global softmax over S before split (default off for relationformer).",
    )
    p.add_argument("--save_individual", action="store_true")
    p.add_argument("--save_input_image", action="store_true")

    p.add_argument("--stats_size", type=int, default=64)
    p.add_argument("--quadrant_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=10)
    return p


def main():
    args = get_parser().parse_args()
    rank, world_size, _local_rank, device = setup_distributed(args.device)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[start] mode={args.mode} world_size={world_size} output={args.output_dir}")

    config = build_config(args)
    model = build_model_and_load(config, args.checkpoint, device)
    num_layers = int(model.transformer.decoder.num_layers)
    selected_layer = num_layers if int(args.decoder_layer) < 1 else int(args.decoder_layer)
    capture = DecoderLayerCapture(model, selected_layer)
    if rank == 0:
        print(f"[capture] decoder_layer={selected_layer}/{num_layers} (1-based)")
    agg = Aggregator(
        stats_size=int(args.stats_size),
        device=device,
        trigger_patch_size=int(args.trigger_patch_size),
        trigger_margin=int(args.trigger_margin),
    )

    try:
        if args.mode == "single":
            run_single(args, model, device, capture, agg, rank)
        else:
            run_dataset(args, config, model, device, capture, agg, rank, world_size)

        agg.sync()
        if rank == 0:
            summary = save_summary(args.output_dir, agg)
            print("[done] summary:", json.dumps(summary, ensure_ascii=False))
    finally:
        capture.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
