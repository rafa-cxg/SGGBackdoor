#!/usr/bin/env python3
import argparse
import json
import os
import sys
from contextlib import contextmanager
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

# Make project imports work when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import util.misc as utils
from datasets import build_dataset
from datasets.coco import make_coco_transforms
from main import build_model_main, get_args_parser
from util.slconfig import SLConfig
from util.utils import to_device

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_trigger_to_pil(image_pil, enabled=False, patch_size=32, margin=10):
    if not enabled:
        return image_pil
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    img = image_pil.copy()
    w, h = img.size
    ps = int(min(patch_size, w, h))
    if ps <= 0:
        return img

    x0 = max(0, min(margin, w - ps))
    y0 = max(0, h - margin - ps)
    x1 = x0 + ps
    y1 = y0 + ps

    arr = np.array(img, dtype=np.uint8)
    arr[y0:y1, x0:x1, 0] = 0
    arr[y0:y1, x0:x1, 1] = 255
    arr[y0:y1, x0:x1, 2] = int(round(0.3 * 255))
    return Image.fromarray(arr, mode="RGB")


def preprocess_caption(caption: str) -> str:
    text = caption.lower().strip()
    if text.endswith("."):
        return text
    return text + "."


def load_name_mappings(dataset_file, data_path):
    if dataset_file == "vg":
        dict_file = os.path.join(
            data_path, "visual_genome", "stanford_filtered", "VG-SGG-dicts.json"
        )
        if not os.path.exists(dict_file):
            raise FileNotFoundError(f"VG dict file not found: {dict_file}")
        with open(dict_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        idx_to_label = payload["idx_to_label"]
        idx_to_predicate = payload["idx_to_predicate"]

        name2classes = {}
        for k, name in sorted(idx_to_label.items(), key=lambda kv: int(kv[0])):
            idx = int(k)
            if name == "__background__":
                continue
            name2classes[name] = idx

        name2predicates = {}
        for k, name in sorted(idx_to_predicate.items(), key=lambda kv: int(kv[0])):
            idx = int(k)
            if name == "__background__":
                name = "[UNK]"
            name2predicates[name] = idx
        return name2classes, name2predicates

    if dataset_file == "coco":
        from datasets.coco import coco_names

        name2classes = {v: int(k) for k, v in coco_names.items()}
        return name2classes, {}

    raise ValueError(f"Unsupported dataset_file for single-image prompt build: {dataset_file}")


def build_text_prompts(name2classes, name2predicates):
    obj_caption = preprocess_caption(". ".join(name2classes.keys()))
    rel_names = [name for name, idx in name2predicates.items() if idx > 0]
    rel_caption = preprocess_caption(". ".join(rel_names)) if rel_names else ""
    return obj_caption, rel_caption


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj:
            return checkpoint_obj["model"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if "net" in checkpoint_obj:
            return checkpoint_obj["net"]
    return checkpoint_obj


def denorm_image_tensor_to_uint8(tensor, mean=MEAN, std=STD):
    if hasattr(tensor, "tensors"):
        tensor = tensor.tensors
    t = tensor.detach().cpu().clone()
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(t.shape)}")
    if t.shape[0] in (1, 3):
        t = t.float()
        for i, (m, s) in enumerate(zip(mean, std)):
            t[i] = t[i] * s + m
        img = t.permute(1, 2, 0).numpy()
    elif t.shape[2] in (1, 3):
        img = t.numpy()
    else:
        raise ValueError(f"Unexpected image shape: {tuple(t.shape)}")
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return img


class QueryAttentionCollector:
    def __init__(
        self,
        output_dir: str,
        stats_size: int,
        device: torch.device,
        target_decoder_layer: int,
        save_individual: bool,
        save_input_image: bool,
        mode: str,
        quadrant_ratio: float,
        trigger_size: int,
        trigger_margin: int,
    ):
        self.output_dir = output_dir
        self.per_image_dir = os.path.join(output_dir, "per_image")
        self.stats_dir = os.path.join(output_dir, "stats")
        os.makedirs(self.per_image_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        self.stats_size = int(stats_size)
        self.device = device
        self.target_decoder_layer = int(target_decoder_layer)
        self.save_individual = bool(save_individual)
        self.save_input_image = bool(save_input_image)
        self.mode = mode
        self.quadrant_ratio = float(quadrant_ratio)
        self.trigger_size = int(trigger_size)
        self.trigger_margin = int(trigger_margin)
        self.topk_ratio = 0.002
        self.topk_label = "0_2_percent"

        self.sum_map = torch.zeros(
            (self.stats_size, self.stats_size), dtype=torch.float32, device=self.device
        )
        self.count = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_mass_sum = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_metric_count = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.trigger_box_xyxy_01_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)

        self.current_key: Optional[str] = None
        self.current_meta = {}
        self.current_consumed = False

    def set_current_sample(self, dataset_idx: int, image_id: int, image_hw: Optional[tuple] = None):
        self.current_key = f"idx{int(dataset_idx):06d}_id{int(image_id):06d}"
        self.current_meta = {
            "dataset_idx": int(dataset_idx),
            "image_id": int(image_id),
            "image_hw": image_hw,
        }
        self.current_consumed = False

    def set_current_single(self, image_path: str, image_hw: Optional[tuple] = None):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        self.current_key = f"single_{stem}"
        self.current_meta = {"dataset_idx": -1, "image_id": -1, "image_hw": image_hw}
        self.current_consumed = False

    def save_input(self, samples):
        if not self.save_input_image or self.current_key is None:
            return
        img = denorm_image_tensor_to_uint8(samples)
        out = os.path.join(self.per_image_dir, f"{self.current_key}_input.png")
        Image.fromarray(img).save(out)

    def build_save_prefix(self):
        if self.current_key is None:
            return os.path.join(self.per_image_dir, "unknown")
        return os.path.join(self.per_image_dir, f"{self.mode}_{self.current_key}")

    def should_capture_layer(self, layer_id: int) -> bool:
        return (
            self.current_key is not None
            and (layer_id == self.target_decoder_layer)
            and (not self.current_consumed)
        )

    def consume_fused(self, fused_map: Optional[torch.Tensor]):
        if fused_map is None or self.current_consumed:
            return
        x = fused_map.detach().float()
        x = x.clamp_min(0.0)
        x_sum = x.sum()
        if float(x_sum.item()) <= 0.0:
            self.current_consumed = True
            return
        x = x / (x_sum + 1e-12)
        x = F.interpolate(
            x[None, None],
            size=(self.stats_size, self.stats_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        x = x.clamp_min(0.0)
        x = x / (x.sum() + 1e-12)
        self.sum_map += x.to(self.sum_map.device)
        self.count += 1.0
        self._accumulate_trigger_metrics(x)

        if self.save_individual and self.current_key is not None:
            out_npy = os.path.join(self.per_image_dir, f"{self.current_key}_fused_resized.npy")
            np.save(out_npy, x.detach().cpu().numpy())
        self.current_consumed = True

    def _accumulate_trigger_metrics(self, x: torch.Tensor):
        image_hw = self.current_meta.get("image_hw", None) if isinstance(self.current_meta, dict) else None
        if not image_hw:
            return
        h_img, w_img = int(image_hw[0]), int(image_hw[1])
        if h_img <= 0 or w_img <= 0:
            return

        ps = int(max(1, min(self.trigger_size, h_img, w_img)))
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

    def sync_distributed(self):
        if utils.get_world_size() <= 1:
            return
        dist.all_reduce(self.sum_map, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.trigger_mass_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.trigger_metric_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.trigger_box_xyxy_01_sum, op=dist.ReduceOp.SUM)

    def _compute_stats(self, mean_map: np.ndarray):
        total = float(mean_map.sum())
        if total <= 0.0:
            return {}
        p = mean_map / total

        flat = p.reshape(-1)
        k_top = max(1, int(np.ceil(self.topk_ratio * flat.size)))
        topk_mass = float(np.partition(flat, -k_top)[-k_top:].sum())

        trigger_stats = {
            "position": "bottom_left",
            "margin_px": int(self.trigger_margin),
            "patch_size_px": int(self.trigger_size),
            "num_images_with_trigger_metrics": int(round(float(self.trigger_metric_count.item()))),
        }
        if float(self.trigger_metric_count.item()) > 0:
            c = self.trigger_metric_count.clamp_min(1.0)
            mean_box = (self.trigger_box_xyxy_01_sum / c).detach().cpu().numpy()
            trigger_stats.update(
                {
                    "trigger_box_xyxy_01_mean": {
                        "x0": float(mean_box[0]),
                        "y0": float(mean_box[1]),
                        "x1": float(mean_box[2]),
                        "y1": float(mean_box[3]),
                    },
                    "trigger_mass_mean": float((self.trigger_mass_sum / c).item()),
                }
            )

        return {
            "topk_mass": {
                f"top_{self.topk_label}": topk_mass,
                f"k_pixels_top_{self.topk_label}": int(k_top),
            },
            "trigger_region": trigger_stats,
        }

    def save_summary(self):
        n = int(round(float(self.count.item())))
        if n <= 0:
            return {"num_images_captured": 0}

        mean_map = (self.sum_map / self.count.clamp_min(1.0)).detach().cpu().numpy()
        mean_map = np.clip(mean_map, 0.0, None)
        mean_map = mean_map / (mean_map.sum() + 1e-12)

        np.save(os.path.join(self.stats_dir, "mean_attention.npy"), mean_map)

        plt.figure(figsize=(5, 4))
        plt.imshow(mean_map, cmap="YlOrBr", interpolation="nearest")
        plt.title(f"Mean attention ({n} images)")
        plt.axis("off")
        plt.colorbar(fraction=0.04, pad=0.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, "mean_attention.png"), dpi=220, bbox_inches="tight")
        plt.close()

        payload = {
            "num_images_captured": n,
            "stats_size": self.stats_size,
            "stats": self._compute_stats(mean_map),
        }
        with open(os.path.join(self.stats_dir, "attention_stats.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload


def resolve_checkpoint_path(args):
    for p in [args.checkpoint, args.resume, args.pretrain_model_path]:
        if p:
            return p
    return ""


def merge_config_to_args(args):
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    if not hasattr(args, "use_ema"):
        args.use_ema = False
    if not hasattr(args, "debug"):
        args.debug = False


@contextmanager
def patched_query_visualizer(collector: QueryAttentionCollector):
    from models.GroundingDINO.transformer import DeformableTransformerDecoderLayer

    original = DeformableTransformerDecoderLayer.cross_attn_heatmaps

    @torch.no_grad()
    def wrapped(self, *args, **kwargs):
        layer_id = int(getattr(self, "layer_id", -1))
        if collector.should_capture_layer(layer_id):
            local_kwargs = dict(kwargs)
            local_kwargs["mode"] = "both"
            local_kwargs["visualize"] = bool(collector.save_individual)
            save_prefix = collector.build_save_prefix()
            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
            local_kwargs["save_prefix"] = save_prefix

            per_level, fused = original(self, *args, **local_kwargs)
            collector.consume_fused(fused)
            return per_level, fused
        return original(self, *args, **kwargs)

    DeformableTransformerDecoderLayer.cross_attn_heatmaps = wrapped
    try:
        yield
    finally:
        DeformableTransformerDecoderLayer.cross_attn_heatmaps = original


def build_runtime_parser():
    parser = argparse.ArgumentParser(
        "Query attention visualization & statistics",
        parents=[get_args_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--mode", choices=["single", "eval"], default="eval")
    parser.add_argument("--checkpoint", default="", help="Model checkpoint to load.")
    parser.add_argument("--image", default="", help="Single-image mode input path.")
    parser.add_argument("--max_images", type=int, default=100, help="Eval mode: number of images to process.")
    parser.add_argument("--stats_size", type=int, default=64, help="Average map output size.")
    parser.add_argument("--vis_output_dir", default="", help="Output root. Default: {output_dir}/query_attention_vis")
    parser.add_argument("--target_decoder_layer", type=int, default=-1, help="Which decoder layer to collect. -1 means last layer.")
    parser.add_argument("--save_individual", action="store_true", help="Save per-image heatmaps.")
    parser.add_argument("--save_input_image", action="store_true", help="Save denormalized input image for each sample.")
    parser.add_argument("--backdoor_eval", action="store_true", help="Eval mode: use backdoor_eval dataset branch.")
    parser.add_argument(
        "--quadrant_ratio",
        type=float,
        default=1.0,
        help="Quadrant stats coverage ratio in [0.05,1.0], centered crop; 1.0 means full map.",
    )
    parser.add_argument("--trigger_image", action="store_true", help="Single mode: add trigger patch before transform.")
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=32,
        help="Trigger patch size in px (used by single-image injection and trigger-region stats).",
    )
    parser.add_argument(
        "--trigger_margin",
        type=int,
        default=10,
        help="Trigger margin in px from image edge (used by single-image injection and trigger-region stats).",
    )
    return parser


def prepare_args():
    parser = build_runtime_parser()
    args = parser.parse_args()

    merge_config_to_args(args)
    args.query_visualization = True
    args.eval = True

    if args.vis_output_dir:
        output_dir = args.vis_output_dir
    else:
        output_dir = os.path.join(args.output_dir, "query_attention_vis")
    args.vis_output_dir = output_dir

    if args.target_decoder_layer < 0:
        args.target_decoder_layer = int(args.dec_layers) - 1
    return args


def setup_distributed_and_device(args):
    utils.init_distributed_mode(args)
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def build_model_and_load_weights(args, device):
    model, _, _ = build_model_main(args)
    ckpt_path = resolve_checkpoint_path(args)
    if not ckpt_path:
        raise ValueError("No checkpoint is provided. Please set --checkpoint (or --resume/--pretrain_model_path).")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_obj)
    missing, unexpected = model.load_state_dict(utils.clean_state_dict(state_dict), strict=False)
    model = model.to(device).eval()

    if utils.is_main_process():
        print(f"[load] checkpoint: {ckpt_path}")
        print(f"[load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return model


@torch.no_grad()
def run_single_image(args, model, collector: QueryAttentionCollector, device):
    if not args.image:
        raise ValueError("--image is required in --mode single")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    image_pil = Image.open(args.image).convert("RGB")
    image_pil = apply_trigger_to_pil(
        image_pil,
        enabled=args.trigger_image,
        patch_size=args.trigger_size,
        margin=args.trigger_margin,
    )

    transforms = make_coco_transforms("val", fix_size=getattr(args, "fix_size", False), args=args)
    image_tensor, _ = transforms(image_pil, None)
    samples = utils.nested_tensor_from_tensor_list([image_tensor])
    image_hw = (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))

    collector.set_current_single(args.image, image_hw=image_hw)
    collector.save_input(samples)

    samples = samples.to(device)
    name2classes, name2predicates = load_name_mappings(args.dataset_file, args.data_path)
    obj_caption, rel_caption = build_text_prompts(name2classes, name2predicates)
    model_kwargs = {"captions": [obj_caption]}
    if rel_caption:
        model_kwargs["rel_captions"] = [rel_caption]

    with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
        _ = model(samples, **model_kwargs)


@torch.no_grad()
def run_eval_subset(args, model, collector: QueryAttentionCollector, device):
    image_set = "test" if getattr(args, "use_test_set", False) else "val"
    # VG backdoor eval path expects backdoor_transforms to be built.
    # In the main program this corresponds to args.backdoor_training=True plus backdoor_eval=True.
    dataset_backdoor_training = bool(getattr(args, "backdoor_training", False) or args.backdoor_eval)
    if bool(args.backdoor_eval) and (not getattr(args, "backdoor_training", False)) and utils.is_main_process():
        print("[info] --backdoor_eval is enabled; dataset build will force backdoor_training=True for transforms.")

    orig_backdoor_training = getattr(args, "backdoor_training", False)
    try:
        # NOTE: build_vg checks args.backdoor_training directly.
        if bool(args.backdoor_eval):
            args.backdoor_training = dataset_backdoor_training

        dataset = build_dataset(
            image_set=image_set,
            args=args,
            backdoor_training=dataset_backdoor_training,
            backdoor_eval=bool(args.backdoor_eval),
        )
    finally:
        args.backdoor_training = orig_backdoor_training

    total = min(int(args.max_images), len(dataset))
    all_indices = list(range(total))
    rank = utils.get_rank()
    world = utils.get_world_size()
    rank_indices = all_indices[rank::world]
    subset = Subset(dataset, rank_indices)

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    need_tgt_for_training = bool(getattr(args, "use_dn", False))
    if bool(getattr(args, "use_text_labels", False)):
        need_tgt_for_training = True

    if utils.is_main_process():
        print(f"[eval] dataset={image_set}, total_selected={total}, world_size={world}")

    for local_i, (samples, targets) in enumerate(loader):
        dataset_idx = rank_indices[local_i]
        image_id = int(targets[0].get("image_id", -1))
        if "size" in targets[0] and torch.is_tensor(targets[0]["size"]) and targets[0]["size"].numel() >= 2:
            h_img = int(targets[0]["size"][0].item())
            w_img = int(targets[0]["size"][1].item())
        else:
            h_img = int(samples.tensors.shape[-2])
            w_img = int(samples.tensors.shape[-1])
        collector.set_current_sample(dataset_idx, image_id, image_hw=(h_img, w_img))
        collector.save_input(samples)

        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            if need_tgt_for_training:
                _ = model(samples, targets)
            else:
                _ = model(samples)


def main():
    args = prepare_args()
    device = setup_distributed_and_device(args)

    os.makedirs(args.vis_output_dir, exist_ok=True)
    collector = QueryAttentionCollector(
        output_dir=args.vis_output_dir,
        stats_size=args.stats_size,
        device=device,
        target_decoder_layer=args.target_decoder_layer,
        save_individual=args.save_individual,
        save_input_image=args.save_input_image,
        mode=args.mode,
        quadrant_ratio=args.quadrant_ratio,
        trigger_size=args.trigger_size,
        trigger_margin=args.trigger_margin,
    )

    model = build_model_and_load_weights(args, device)

    with patched_query_visualizer(collector):
        if args.mode == "single":
            if utils.get_rank() == 0:
                run_single_image(args, model, collector, device)
        else:
            run_eval_subset(args, model, collector, device)

    if args.distributed:
        dist.barrier()
    collector.sync_distributed()

    if utils.is_main_process():
        summary = collector.save_summary()
        print(f"[done] output_dir: {args.vis_output_dir}")
        print(f"[done] summary: {json.dumps(summary, ensure_ascii=False)}")

    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
