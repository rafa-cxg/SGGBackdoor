import argparse
import ast
import json
import os
import sys
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import FancyArrowPatch

# Make `util.*` importable when this script is run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.visualizer import COCOVisualizer, renorm
import util.misc as utils
from util.box_ops import box_xyxy_to_cxcywh

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

mpl.rcParams["pdf.fonttype"] = 42


def save_side_by_side_visualization(
    vis,
    img,
    tgt,
    relations=None,
    dpi=300,
    savedir=None,
    show_in_console=True,
    title=None,
    save_name=None,
):
    img_denorm = renorm(img).permute(1, 2, 0).cpu().numpy()
    img_h, img_w = img_denorm.shape[:2]

    if relations is None:
        fig, ax_img = plt.subplots(1, 1, figsize=(img_w / 90.0, img_h / 90.0), dpi=dpi)
        ax_graph = None
    else:
        graph_width_ratio = 1.38
        image_width_ratio = max(1.0, img_w / max(img_h, 1))
        fig_h = max(6.0, min(14.0, img_h / 90.0))
        fig_w = fig_h * (image_width_ratio + graph_width_ratio)
        fig, (ax_img, ax_graph) = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            dpi=max(dpi, 320),
            constrained_layout=False,
            gridspec_kw={
                "width_ratios": [image_width_ratio, graph_width_ratio],
                "wspace": 0.006,
            },
        )
        fig.subplots_adjust(left=0.0, right=1.0, top=0.985, bottom=0.024, wspace=0.006)

    plt.rcParams["font.size"] = "8"

    ax_img.imshow(img_denorm)
    tgt_to_draw = vis._filter_tgt_by_relations(tgt, relations)
    vis.addtgt(ax_img, tgt_to_draw, None)
    ax_img.set_axis_off()
    ax_img.set_aspect("equal", adjustable="box")

    if relations is not None and ax_graph is not None:
        G = nx.DiGraph()
        edges = [tuple(r[:2]) for r in relations]
        nodes = list(np.unique(np.array(edges, dtype=object)))
        labeldict = {node: node for node in nodes}
        edge_labeldict = {
            (edge[0], edge[1]): vis._resolve_predicate_label(edge[2], tgt)
            for edge in relations
        }

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        pos = vis._bbox_relative_layout(nodes, tgt, scale=0.72)
        if pos is None:
            pos = vis._simple_ring_layout(nodes, G, scale=0.36)

        node_size = [max(1350, len(str(node)) * 215) for node in labeldict]
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax_graph,
            node_size=node_size,
            node_color="lightcoral",
            edgecolors="black",
            linewidths=1.2,
            alpha=0.95,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax_graph,
            labels=labeldict,
            font_size=11,
            font_weight="bold",
        )

        edge_colors = [
            "#8e44ad",
            "#16a085",
            "#2980b9",
            "#d35400",
            "#c0392b",
            "#2c3e50",
            "#7f8c8d",
            "#27ae60",
            "#8e5b3a",
            "#5b2c6f",
        ]
        edges_list = sorted(list(G.edges()), key=lambda e: (str(e[0]), str(e[1])))
        reverse_set = set((v, u) for (u, v) in edges_list)
        out_neighbor_rank = {}
        for u in G.nodes():
            neigh = sorted(list(G.successors(u)), key=lambda x: str(x))
            out_neighbor_rank[u] = {v: i for i, v in enumerate(neigh)}

        for idx, (u, v) in enumerate(edges_list):
            color = edge_colors[idx % len(edge_colors)]
            p0 = np.asarray(pos[u], dtype=float)
            p1 = np.asarray(pos[v], dtype=float)
            vec = p1 - p0
            norm = np.linalg.norm(vec) + 1e-6
            tangent = vec / norm
            normal = np.array([-tangent[1], tangent[0]])

            has_reverse = (u, v) in reverse_set and (v, u) in reverse_set
            if has_reverse:
                rad = 0.24 if str(u) < str(v) else -0.24
            else:
                rank = out_neighbor_rank.get(u, {}).get(v, 0)
                deg = max(1, G.out_degree(u))
                center = 0.5 * (deg - 1)
                rad = 0.10 + 0.07 * (rank - center)
                rad = float(np.clip(rad, -0.34, 0.34))

            arrow = FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=22,
                lw=2.3,
                color=color,
                alpha=0.95,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=18,
                shrinkB=20,
            )
            ax_graph.add_patch(arrow)

            lbl = edge_labeldict.get((u, v), "")
            candidates = []
            for t in (0.70, 0.62, 0.78):
                base = p0 * (1.0 - t) + p1 * t
                for sgn in (1.0, -1.0):
                    off = normal * sgn * (0.035 + abs(rad) * 0.045)
                    candidates.append(base + off)

            node_points = [np.asarray(pos[n], dtype=float) for n in G.nodes()]
            if len(node_points) > 0:
                node_points = np.stack(node_points, axis=0)
                best_pt = max(
                    candidates,
                    key=lambda p: float(np.min(np.linalg.norm(node_points - p[None, :], axis=1))),
                )
            else:
                best_pt = candidates[0]

            ax_graph.text(
                best_pt[0],
                best_pt[1],
                str(lbl),
                fontsize=12,
                color=color,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9, pad=0.2),
            )

        if len(pos) > 0:
            pts = np.stack([np.asarray(pos[n], dtype=float) for n in G.nodes()], axis=0)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            span_x = float(x_max - x_min)
            span_y = float(y_max - y_min)
            target_aspect = img_w / max(img_h, 1)

            if span_y < 1e-3:
                span_y = 1.0
            if span_x < 1e-3:
                span_x = 1.0

            graph_aspect = span_x / span_y
            if graph_aspect < target_aspect:
                span_x = target_aspect * span_y
            else:
                span_y = span_x / target_aspect

            pad_x = max(0.01, 0.018 * span_x)
            pad_y = max(0.04, 0.065 * span_y)
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            ax_graph.set_xlim(cx - 0.5 * span_x - pad_x, cx + 0.5 * span_x + pad_x)
            ax_graph.set_ylim(cy - 0.5 * span_y - pad_y, cy + 0.5 * span_y + pad_y)

        ax_graph.set_aspect("equal", adjustable="box")
        ax_graph.set_axis_off()

    if title:
        fig.suptitle(title, y=0.995)

    if show_in_console:
        plt.show()

    if savedir is not None:
        if save_name is not None:
            ext = os.path.splitext(save_name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
                save_name = save_name + ".pdf"
            savename = os.path.join(savedir, save_name)
        elif "image_id" in tgt:
            savename = os.path.join(savedir, f"{int(tgt['image_id'])}.jpg")
        else:
            savename = os.path.join(savedir, "visualization.pdf")
        print(f"savename: {savename}")
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, bbox_inches="tight", pad_inches=0.004)
    plt.close(fig)


def parse_literal(value, name):
    try:
        return ast.literal_eval(value)
    except Exception as exc:
        raise ValueError(f"Failed to parse --{name}: {value}") from exc


def load_inputs(args):
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "boxes" in payload:
            boxes = payload["boxes"]
        elif "boxes_norm_cxcywh" in payload:
            boxes = payload["boxes_norm_cxcywh"]
        else:
            raise KeyError("input_json must contain `boxes` or `boxes_norm_cxcywh`")
        labels = payload.get("labels")
        relations = payload.get("relations")
        title = payload.get("title", args.title)
        caption = payload.get("caption")
        image_from_json = payload.get("image", "")
    else:
        boxes = parse_literal(args.boxes, "boxes")
        labels = parse_literal(args.labels, "labels") if args.labels else None
        relations = parse_literal(args.relations, "relations") if args.relations else None
        title = args.title
        caption = None
        image_from_json = ""

    return boxes, labels, relations, title, caption, image_from_json


def infer_output_name(args):
    stem = args.image_name.strip() if args.image_name else os.path.splitext(os.path.basename(args.image))[0]
    if args.auto_from_model and not args.input_json:
        cfg_name = os.path.basename(args.config_file).lower()
        suffix = "_poison" if "backdoor" in cfg_name else "_clean"
        stem = f"{stem}{suffix}"
    return stem


def apply_trigger_to_pil(image_pil, enabled=False, patch_size=32, margin=10):
    """Insert a fixed-color trigger patch at bottom-left before tensor conversion."""
    if not enabled:
        return image_pil
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    img = image_pil.copy()
    w, h = img.size
    ps = int(min(patch_size, w, h))
    if ps <= 0:
        return img

    # Bottom-left with margin from borders.
    x0 = max(0, min(margin, w - ps))
    y0 = max(0, h - margin - ps)
    x1 = x0 + ps
    y1 = y0 + ps

    arr = np.array(img, dtype=np.uint8)
    # patch_trigger[0]=0.0, patch_trigger[1]=1.0, patch_trigger[2]=0.3
    arr[y0:y1, x0:x1, 0] = 0
    arr[y0:y1, x0:x1, 1] = 255
    arr[y0:y1, x0:x1, 2] = int(round(0.3 * 255))
    return Image.fromarray(arr, mode="RGB")


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

        name2classes = OrderedDict()
        class2name = {}
        for k, name in sorted(idx_to_label.items(), key=lambda kv: int(kv[0])):
            idx = int(k)
            if name == "__background__":
                continue
            name2classes[name] = idx
            class2name[idx] = name

        name2predicates = OrderedDict()
        predicate2name = {}
        for k, name in sorted(idx_to_predicate.items(), key=lambda kv: int(kv[0])):
            idx = int(k)
            if name == "__background__":
                name = "[UNK]"
            name2predicates[name] = idx
            predicate2name[idx] = name
        return name2classes, class2name, name2predicates, predicate2name

    if dataset_file == "coco":
        from datasets.coco import coco_names
        name2classes = OrderedDict((v, int(k)) for k, v in coco_names.items())
        class2name = {v: k for k, v in name2classes.items()}
        return name2classes, class2name, OrderedDict(), {}

    raise ValueError(f"Unsupported dataset_file for standalone infer: {dataset_file}")


def preprocess_caption(caption):
    text = caption.lower().strip()
    if text.endswith("."):
        return text
    return text + "."


def build_text_prompts(name2classes, name2predicates):
    obj_caption = preprocess_caption(". ".join(name2classes.keys()))
    rel_names = [name for name, idx in name2predicates.items() if idx > 0]
    rel_caption = preprocess_caption(". ".join(rel_names)) if rel_names else ""
    return obj_caption, rel_caption


def pad_predicate_index_space(name2predicates):
    """
    Keep graph_infer unchanged by ensuring len(mapping) > max(index).
    This is only for standalone visualization path.
    """
    if not name2predicates:
        return name2predicates
    padded = OrderedDict(name2predicates)
    used = set(int(v) for v in padded.values())
    max_idx = max(used)
    if len(padded) > max_idx:
        return padded

    # Fill missing low indices (typically missing bg=0) with dummy names.
    for idx in range(max_idx + 1):
        if len(padded) > max_idx:
            break
        if idx not in used:
            dummy = f"__pad_predicate_{idx}__"
            while dummy in padded:
                dummy = "_" + dummy
            padded[dummy] = idx
            used.add(idx)
    return padded


def build_model_args(args):
    from util.slconfig import SLConfig
    from main import get_args_parser
    parser = get_args_parser()
    argv = [
        "--config_file",
        args.config_file,
        "--output_dir",
        args.output_dir,
        "--dataset_file",
        args.dataset_file,
        "--data_path",
        args.data_path,
        "--device",
        args.device,
        "--num_workers",
        "0",
        "--trigger_object_idx",
        str(args.trigger_object_idx),
        "--trigger_predicate_idx",
        str(args.trigger_predicate_idx),
    ]
    if args.model_options:
        argv += ["--options"] + args.model_options
    model_args = parser.parse_args(argv)

    cfg = SLConfig.fromfile(model_args.config_file)
    if model_args.options is not None:
        cfg.merge_from_dict(model_args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(model_args, k):
            setattr(model_args, k, v)
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    return model_args


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj:
            return checkpoint_obj["model"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if "net" in checkpoint_obj:
            return checkpoint_obj["net"]
    return checkpoint_obj


def run_model_infer(args):
    from datasets.coco import make_coco_transforms
    from main import build_model_main
    model_args = build_model_args(args)
    name2classes, class2name, name2predicates, predicate2name = load_name_mappings(
        model_args.dataset_file, model_args.data_path
    )
    name2predicates_for_infer = pad_predicate_index_space(name2predicates)

    model, _, postprocessors = build_model_main(model_args)
    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_obj)
    missing, unexpected = model.load_state_dict(utils.clean_state_dict(state_dict), strict=False)
    if args.verbose_model_load:
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    device = torch.device(model_args.device)
    model = model.to(device).eval()
    postprocessors["bbox"].rln_proj = getattr(model, "rln_proj", None)
    postprocessors["bbox"].rln_classifier = getattr(model, "rln_classifier", None)
    postprocessors["bbox"].rln_freq_bias = getattr(model, "rln_freq_bias", None)
    postprocessors["bbox"].name2classes = name2classes
    if name2predicates_for_infer:
        postprocessors["bbox"].name2predicates = name2predicates_for_infer

    image_pil = Image.open(args.image).convert("RGB")
    image_pil = apply_trigger_to_pil(
        image_pil,
        enabled=args.trigger_image,
        patch_size=args.trigger_size,
        margin=args.trigger_margin,
    )
    orig_w, orig_h = image_pil.size
    transforms = make_coco_transforms("val", fix_size=getattr(model_args, "fix_size", False), args=model_args)
    image_tensor, _ = transforms(image_pil, None)
    samples = utils.nested_tensor_from_tensor_list([image_tensor]).to(device)

    obj_caption, rel_caption = build_text_prompts(name2classes, name2predicates)
    model_kwargs = {"captions": [obj_caption]}
    if rel_caption:
        model_kwargs["rel_captions"] = [rel_caption]

    with torch.no_grad():
        outputs = model(samples, **model_kwargs)
        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        result = postprocessors["bbox"](outputs, target_sizes)[0]

    boxes_xyxy = result["boxes"].detach().cpu()
    boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=boxes_cxcywh.dtype)
    boxes_norm = boxes_cxcywh / scale

    labels_tensor = result["labels"].detach().cpu()
    scores_tensor = result["scores"].detach().cpu()
    keep = scores_tensor >= args.box_score_threshold
    boxes_norm = boxes_norm[keep]
    boxes_xyxy = boxes_xyxy[keep]
    labels_tensor = labels_tensor[keep]
    scores_tensor = scores_tensor[keep]
    kept_old_idx = np.where(keep.numpy())[0].tolist()

    label_names = [class2name.get(int(x), str(int(x))) for x in labels_tensor.tolist()]

    # Merge highly-overlapped detections (IoU > 0.8):
    # 1) if both labels are in {car, truck}, fuse as one "car";
    # 2) otherwise keep the higher-confidence object.
    merge_set = {"car", "truck"}
    n_kept = len(label_names)
    base_to_fused = {i: i for i in range(n_kept)}
    fused_boxes_norm = boxes_norm
    fused_boxes_xyxy = boxes_xyxy
    fused_scores = scores_tensor
    fused_label_names = list(label_names)
    fused_any = False
    if n_kept > 1:
        parent = list(range(n_kept))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n_kept):
            bi = boxes_xyxy[i]
            ai = max(0.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
            for j in range(i + 1, n_kept):
                bj = boxes_xyxy[j]
                aj = max(0.0, float((bj[2] - bj[0]) * (bj[3] - bj[1])))
                ix1 = max(float(bi[0]), float(bj[0]))
                iy1 = max(float(bi[1]), float(bj[1]))
                ix2 = min(float(bi[2]), float(bj[2]))
                iy2 = min(float(bi[3]), float(bj[3]))
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                union = ai + aj - inter + 1e-6
                iou = inter / union
                if iou > 0.8:
                    _union(i, j)

        groups = {}
        for i in range(n_kept):
            r = _find(i)
            groups.setdefault(r, []).append(i)

        if any(len(v) > 1 for v in groups.values()):
            fused_any = True
            group_items = sorted(groups.values(), key=lambda x: min(x))
            out_boxes_norm = []
            out_boxes_xyxy = []
            out_scores = []
            out_label_names = []
            for new_idx, group in enumerate(group_items):
                # Pick the highest-score box as representative.
                rep = max(group, key=lambda k: float(scores_tensor[k]))
                out_boxes_norm.append(boxes_norm[rep])
                out_boxes_xyxy.append(boxes_xyxy[rep])
                out_scores.append(scores_tensor[rep])

                g_labels = [str(label_names[k]).lower() for k in group]
                # car/truck duplicates are unified to "car";
                # otherwise keep the label of the highest-score representative.
                if len(g_labels) > 0 and all(n in merge_set for n in g_labels):
                    out_label_names.append("car")
                else:
                    out_label_names.append(label_names[rep])

                for old_idx in group:
                    base_to_fused[old_idx] = new_idx

            fused_boxes_norm = torch.stack(out_boxes_norm, dim=0)
            fused_boxes_xyxy = torch.stack(out_boxes_xyxy, dim=0)
            fused_scores = torch.stack(out_scores, dim=0)
            fused_label_names = out_label_names

    boxes_norm = fused_boxes_norm
    boxes_xyxy = fused_boxes_xyxy
    scores_tensor = fused_scores

    node_names = []
    box_labels = []
    for i, (cls_name, score) in enumerate(zip(fused_label_names, scores_tensor.tolist())):
        node_names.append(f"{cls_name}#{i}")
        if args.show_box_score:
            box_labels.append(f"{cls_name}:{score:.2f}")
        else:
            box_labels.append(cls_name)

    relations = []
    watch_hits = []
    graph = result.get("graph")
    if graph is not None:
        all_pairs = graph["all_node_pairs"].detach().cpu().numpy()
        all_rel = graph["all_relation"].detach().cpu().numpy()
        old_to_new = {}
        for new_i, old_i in enumerate(kept_old_idx):
            old_to_new[int(old_i)] = base_to_fused[new_i]

        candidates = []
        for pair, rel_scores in zip(all_pairs, all_rel):
            old_sub, old_obj = int(pair[0]), int(pair[1])
            if old_sub not in old_to_new or old_obj not in old_to_new:
                continue
            new_sub = old_to_new[old_sub]
            new_obj = old_to_new[old_obj]
            if new_sub == new_obj:
                continue
            if rel_scores.shape[0] <= 1:
                continue
            pred_idx = int(np.argmax(rel_scores[1:]) + 1)
            pred_score = float(rel_scores[pred_idx])
            pred_name = predicate2name.get(pred_idx, str(pred_idx))
            candidates.append({
                "sub": node_names[new_sub],
                "obj": node_names[new_obj],
                "pred_idx": pred_idx,
                "pred_name": pred_name,
                "score": pred_score,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_by_threshold = [c for c in candidates if c["score"] >= args.relation_score_threshold]
        top_main = top_by_threshold[: args.relation_topk]

        if args.watch_predicate_idx > 0:
            top_watch = candidates[: args.watch_topn]
            watch_hits = [
                c for c in top_watch
                if c["pred_idx"] == args.watch_predicate_idx
                and c["score"] >= args.relation_score_threshold
            ]
            if len(watch_hits) > 0:
                print(
                    f"[watch] predicate_idx={args.watch_predicate_idx} "
                    f"hits={len(watch_hits)} in top{args.watch_topn}"
                )

        # Merge watch hits without consuming relation_topk quota.
        merged = list(top_main)
        seen = {(m["sub"], m["obj"], m["pred_idx"]) for m in merged}
        for hit in watch_hits:
            key = (hit["sub"], hit["obj"], hit["pred_idx"])
            if key not in seen:
                merged.append(hit)
                seen.add(key)

        for c in merged:
            rel_text = f'{c["pred_name"]}:{c["score"]:.2f}' if args.show_relation_score else c["pred_name"]
            relations.append([c["sub"], c["obj"], rel_text])

    dump_payload = {
        "image": args.image,
        "image_name": args.output_name,
        "boxes_norm_cxcywh": boxes_norm.tolist(),
        "boxes_abs_xyxy": boxes_xyxy.tolist(),
        "labels": fused_label_names,
        "scores": scores_tensor.tolist(),
        "relations": relations,
        "car_truck_fused": bool(fused_any),
        "watch_predicate_idx": args.watch_predicate_idx,
        "watch_topn": args.watch_topn,
        "watch_hits_topn": [
            {
                "sub": c["sub"],
                "obj": c["obj"],
                "pred_idx": c["pred_idx"],
                "pred_name": c["pred_name"],
                "score": c["score"],
            }
            for c in watch_hits
        ],
    }

    if args.dump_json:
        os.makedirs(os.path.dirname(args.dump_json), exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(dump_payload, f, indent=2, ensure_ascii=False)

    return (
        boxes_norm.tolist(),
        box_labels,
        relations,
        args.output_name,
        args.title,
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="", help="Path to an RGB image; optional when input_json contains `image`")
    parser.add_argument("--image_name", default="",
                        help="Output jpg stem (without extension). Default: input image filename stem")
    parser.add_argument("--auto_from_model", action="store_true",
                        help="Infer boxes/relations from OvSGTR model using config+checkpoint")
    parser.add_argument("--config_file", default="", help="OvSGTR config path")
    parser.add_argument("--checkpoint", default="", help="OvSGTR checkpoint path")
    parser.add_argument("--dataset_file", default="vg", choices=["vg", "coco"])
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="./debug_vis/standalone/tmp")
    parser.add_argument("--model_options", nargs="*", default=[],
                        help="Extra OvSGTR options in key=value format")
    parser.add_argument("--trigger_object_idx", type=int, default=26)
    parser.add_argument("--trigger_predicate_idx", type=int, default=23)
    parser.add_argument("--box_score_threshold", type=float, default=0.2)
    parser.add_argument("--show_box_score", action="store_true",
                        help="Show bbox score in object label text (default: off)")
    parser.add_argument("--relation_score_threshold", type=float, default=0.2)
    parser.add_argument("--relation_topk", type=int, default=30)
    parser.add_argument("--show_relation_score", action="store_true")
    parser.add_argument("--watch_predicate_idx", type=int, default=-1,
                        help="If >0, append top-N hits with this predicate idx without using relation_topk quota; watch hits must also pass relation_score_threshold")
    parser.add_argument("--watch_topn", type=int, default=20,
                        help="Top-N relations to scan for watch_predicate_idx")
    parser.add_argument("--dump_json", default="./debug_vis/standalone/auto_result.json")
    parser.add_argument("--verbose_model_load", action="store_true")
    parser.add_argument("--input_json", default="", help="Optional json file with keys: boxes, labels, relations")
    parser.add_argument("--boxes", default="[[0.30,0.55,0.28,0.45],[0.62,0.58,0.38,0.50]]",
                        help="List of boxes in normalized xywh format")
    parser.add_argument("--labels", default="['person','bicycle']", help="List of box labels")
    parser.add_argument("--relations", default="[[0,1,'riding']]",
                        help="List of relations: [sub_idx, obj_idx, predicate]")
    parser.add_argument("--title", default="")
    parser.add_argument("--savedir", default="./debug_vis/standalone")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--trigger_image", action="store_true",
                        help="Insert a fixed trigger patch before tensor conversion")
    parser.add_argument("--trigger_size", type=int, default=32,
                        help="Trigger patch size in pixels")
    parser.add_argument("--trigger_margin", type=int, default=10,
                        help="Margin to image border in pixels")
    args = parser.parse_args()
    args.output_name = infer_output_name(args)

    if args.input_json:
        if args.auto_from_model:
            print("[visualize_scene_graph_standalone] --input_json is set, skip model inference.")
        boxes, labels, relations, title, caption, image_from_json = load_inputs(args)
        if not args.image:
            args.image = image_from_json
        if not args.image:
            raise ValueError("Please provide --image, or include `image` in --input_json")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_name = args.output_name
    elif args.auto_from_model:
        if not args.image:
            raise ValueError("--image is required when --auto_from_model is enabled")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        if not args.config_file:
            raise ValueError("--config_file is required when --auto_from_model is enabled")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --auto_from_model is enabled")
        boxes, labels, relations, image_name, title, caption = run_model_infer(args)
    else:
        boxes, labels, relations, title, caption, image_from_json = load_inputs(args)
        if not args.image:
            args.image = image_from_json
        if not args.image:
            raise ValueError("Please provide --image, or include `image` in --input_json")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_name = args.output_name

    image = Image.open(args.image).convert("RGB")
    image = apply_trigger_to_pil(
        image,
        enabled=args.trigger_image,
        patch_size=args.trigger_size,
        margin=args.trigger_margin,
    )
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    h, w = image_np.shape[:2]

    # visualizer.renorm() expects normalized tensor and denormalizes internally.
    image_norm = (image_np - MEAN) / STD
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)

    tgt = {
        "image_id": torch.tensor(0),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "size": torch.tensor([h, w], dtype=torch.int64),
    }
    if labels is not None:
        tgt["box_label"] = labels
    if caption:
        tgt["caption"] = caption

    # util.visualizer expects graph node labels to be strings (len() is used).
    if relations is not None:
        normalized_relations = []
        for rel in relations:
            if len(rel) < 3:
                raise ValueError(f"Each relation must be [sub, obj, predicate], got: {rel}")
            sub, obj, pred = rel[0], rel[1], rel[2]
            if isinstance(sub, (int, np.integer)):
                sub = labels[sub] if labels is not None and sub < len(labels) else str(sub)
            if isinstance(obj, (int, np.integer)):
                obj = labels[obj] if labels is not None and obj < len(labels) else str(obj)
            normalized_relations.append([sub, obj, pred])
        relations = normalized_relations

    vis = COCOVisualizer()
    save_side_by_side_visualization(
        vis=vis,
        img=image_tensor,
        tgt=tgt,
        relations=relations,
        savedir=args.savedir,
        show_in_console=not args.no_show,
        title=title,
        dpi=args.dpi,
        save_name=f"{image_name}.pdf",
    )


if __name__ == "__main__":
    main()
