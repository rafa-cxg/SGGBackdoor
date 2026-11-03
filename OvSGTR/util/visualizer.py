# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import re
import torch
import numpy as np
import cv2
import datetime

import copy
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, FancyArrowPatch
from pycocotools import mask as maskUtils
from matplotlib import transforms
import networkx as nx

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        self._colors = None
        pass

    @staticmethod
    def _barycenter_sort(nodes, neigh_dict, ref_index):
        def _key(n):
            neigh = neigh_dict.get(n, [])
            if len(neigh) == 0:
                return (1e9, str(n))
            vals = [ref_index[x] for x in neigh if x in ref_index]
            if len(vals) == 0:
                return (1e9, str(n))
            return (float(np.mean(vals)), str(n))
        return sorted(nodes, key=_key)

    def _simple_ring_layout(self, nodes, G, scale=0.24):
        # Deterministic one-ring layout: equal-angle placement.
        if len(nodes) == 0:
            return {}
        ordered = sorted(nodes, key=lambda n: (-G.degree(n), str(n)))
        # Keep layout compact by default, with slight expansion only for many nodes.
        compact_scale = min(0.28, max(0.20, scale + 0.002 * max(0, len(ordered) - 10)))
        theta = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
        return {
            n: np.array([compact_scale * np.cos(t), compact_scale * np.sin(t)])
            for n, t in zip(ordered, theta)
        }

    def _bbox_relative_layout(self, nodes, tgt, scale=0.24):
        if tgt is None or 'boxes' not in tgt:
            return None

        boxes = tgt['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(boxes)
        if boxes.ndim != 2 or boxes.shape[0] == 0:
            return None

        labels = tgt.get('box_label', None)
        if labels is None:
            labels = []
        labels = [str(x) for x in labels]
        label_to_indices = {}
        for i, lb in enumerate(labels):
            label_to_indices.setdefault(lb, []).append(i)

        def _node_to_idx(node_name):
            # Preferred format: xxx#k
            m = re.search(r'#(\d+)$', str(node_name))
            if m:
                idx = int(m.group(1))
                if 0 <= idx < boxes.shape[0]:
                    return idx
            # If node is an integer id.
            if isinstance(node_name, (int, np.integer)):
                idx = int(node_name)
                if 0 <= idx < boxes.shape[0]:
                    return idx
            # Fallback: try exact label matching.
            idxs = label_to_indices.get(str(node_name), [])
            if len(idxs) > 0:
                return idxs[0]
            return None

        centers = boxes[:, :2]  # normalized cx, cy
        mapped = {}
        for n in nodes:
            idx = _node_to_idx(n)
            if idx is None:
                continue
            cx, cy = centers[idx]
            # Keep only rough relative direction (not strict alignment to bbox).
            x = (float(cx) - 0.5)
            y = (0.5 - float(cy))
            mapped[n] = np.array([x, y], dtype=float)

        # Require high coverage; otherwise fallback to ring layout.
        if len(mapped) < max(1, int(0.7 * len(nodes))):
            return None

        # Angle from bbox-relative direction (rough spatial prior).
        keys = list(mapped.keys())
        pts = np.stack([mapped[k] for k in keys], axis=0)
        ang = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)
        order = np.argsort(ang)
        ordered_keys = [keys[i] for i in order]
        ordered_ang = ang[order]

        n = len(ordered_keys)
        pos = {}
        if n == 1:
            pos[ordered_keys[0]] = np.array([0.0, 0.0], dtype=float)
            return pos

        # Adaptive angular spacing:
        # dense sectors are expanded, sparse sectors compressed, while preserving rough direction.
        gaps_raw = np.diff(np.r_[ordered_ang, ordered_ang[0] + 2 * np.pi])
        uniform_gap = (2 * np.pi) / n
        blend = 0.78
        new_gaps = (1.0 - blend) * gaps_raw + blend * uniform_gap
        new_gaps = np.clip(new_gaps, 0.45 * uniform_gap, 1.9 * uniform_gap)
        new_gaps *= (2 * np.pi) / (new_gaps.sum() + 1e-8)

        theta = np.zeros(n, dtype=float)
        theta[0] = ordered_ang[0]
        for i in range(1, n):
            theta[i] = theta[i - 1] + new_gaps[i - 1]

        # Local radius adapts to neighborhood density:
        # denser area -> slightly larger radius (more arc length),
        # sparse area -> slightly smaller radius (more compact).
        # Use raw (pre-smoothed) local gaps for density estimation so dense sectors
        # are reliably expanded rather than accidentally compressed.
        local_gap = 0.5 * (np.roll(gaps_raw, 1) + gaps_raw)
        density = np.clip(uniform_gap / (local_gap + 1e-8), 0.7, 1.6)
        base_r = scale * 0.74
        r = base_r * np.clip(1.0 + 0.42 * (density - 1.0), 0.78, 1.30)

        for t, rr, k in zip(theta, r, ordered_keys):
            pos[k] = np.array([rr * np.cos(t), rr * np.sin(t)], dtype=float)
        return pos

    def _layered_circular_layout(self, G, scale=0.42):
        if len(G.nodes()) <= 1:
            return nx.circular_layout(G, scale=scale)

        # 1) Collapse strongly connected components to a DAG for robust layering.
        dag = nx.condensation(G)
        comp_of = dag.graph['mapping']  # node -> comp_id

        # 2) Longest-path layering on DAG.
        comp_layer = {}
        for c in nx.topological_sort(dag):
            preds = list(dag.predecessors(c))
            comp_layer[c] = 0 if len(preds) == 0 else max(comp_layer[p] + 1 for p in preds)

        layer_nodes = {}
        for n in G.nodes():
            l = comp_layer[comp_of[n]]
            layer_nodes.setdefault(l, []).append(n)

        max_layer = max(layer_nodes.keys())
        for l in layer_nodes:
            # Stable initialization.
            layer_nodes[l] = sorted(layer_nodes[l], key=lambda x: (-G.degree(x), str(x)))

        # 3) Barycentric sweeps to reduce crossings between neighboring layers.
        for _ in range(4):
            # Forward sweep
            for l in range(1, max_layer + 1):
                prev = layer_nodes.get(l - 1, [])
                ref_idx = {n: i for i, n in enumerate(prev)}
                neigh = {}
                for n in layer_nodes.get(l, []):
                    neigh[n] = list(set(G.predecessors(n)).union(set(G.successors(n))))
                layer_nodes[l] = self._barycenter_sort(layer_nodes[l], neigh, ref_idx)
            # Backward sweep
            for l in range(max_layer - 1, -1, -1):
                nxt = layer_nodes.get(l + 1, [])
                ref_idx = {n: i for i, n in enumerate(nxt)}
                neigh = {}
                for n in layer_nodes.get(l, []):
                    neigh[n] = list(set(G.predecessors(n)).union(set(G.successors(n))))
                layer_nodes[l] = self._barycenter_sort(layer_nodes[l], neigh, ref_idx)

        # 4) Place layers on concentric circles.
        pos = {}
        if max_layer == 0:
            only = layer_nodes[0]
            theta = np.linspace(0, 2 * np.pi, len(only), endpoint=False)
            r = scale * 0.75
            for t, n in zip(theta, only):
                pos[n] = np.array([r * np.cos(t), r * np.sin(t)])
            return pos

        r_min = scale * 0.22
        r_max = scale
        for l in range(max_layer + 1):
            ring = layer_nodes.get(l, [])
            if len(ring) == 0:
                continue
            r = r_min + (r_max - r_min) * (l / max_layer)
            phase = (np.pi / max(1, len(ring))) * (l % 2)
            theta = np.linspace(0, 2 * np.pi, len(ring), endpoint=False) + phase
            for t, n in zip(theta, ring):
                pos[n] = np.array([r * np.cos(t), r * np.sin(t)])
        return pos

    def _relation_object_indices(self, relations):
        keep = set()
        if relations is None:
            return []
        for rel in relations:
            if len(rel) < 2:
                continue
            for endpoint in rel[:2]:
                if isinstance(endpoint, (int, np.integer)):
                    keep.add(int(endpoint))
                    continue
                m = re.search(r'#(\d+)$', str(endpoint))
                if m:
                    keep.add(int(m.group(1)))
        return sorted(keep)

    def _filter_tgt_by_relations(self, tgt, relations):
        if tgt is None or relations is None or len(relations) == 0:
            return tgt
        if 'boxes' not in tgt:
            return tgt

        if isinstance(tgt['boxes'], torch.Tensor):
            n = int(tgt['boxes'].shape[0])
        else:
            n = len(tgt['boxes'])

        keep_idx = [i for i in self._relation_object_indices(relations) if 0 <= i < n]
        if len(keep_idx) == 0:
            return tgt

        out = copy.deepcopy(tgt)
        if isinstance(out['boxes'], torch.Tensor):
            idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
            out['boxes'] = out['boxes'][idx_t]
        else:
            out['boxes'] = [out['boxes'][i] for i in keep_idx]

        if 'box_label' in out:
            if isinstance(out['box_label'], torch.Tensor):
                idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
                out['box_label'] = out['box_label'][idx_t]
            else:
                out['box_label'] = [out['box_label'][i] for i in keep_idx]
        return out

    @staticmethod
    def _resolve_predicate_label(pred, tgt=None):
        """
        Robustly resolve predicate label from either string or index.
        Supports both predicate vocab with background (size=51, idx 0..50)
        and without background (size=50, idx 0..49).
        """
        if not isinstance(pred, (int, np.integer)):
            return str(pred)

        pred_idx = int(pred)
        if tgt is None:
            return str(pred_idx)

        vocab = tgt.get('predicate_names', None)
        if vocab is None:
            vocab = tgt.get('predicates', None)
        if vocab is None:
            return str(pred_idx)

        try:
            vocab_len = len(vocab)
        except Exception:
            return str(pred_idx)

        if vocab_len == 0:
            return str(pred_idx)

        # Direct lookup.
        if 0 <= pred_idx < vocab_len:
            return str(vocab[pred_idx])

        # Compatibility: if vocab excludes background, shift by -1.
        shifted = pred_idx - 1
        if 0 <= shifted < vocab_len:
            return str(vocab[shifted])

        # Safe fallback: no crash.
        return str(pred_idx)

    def visualize(self, img, tgt, caption=None, colors=None, dpi=300, savedir=None, show_in_console=True, title=None, close_axis=False, relations=None, save_name=None):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        if relations is not None:
            # Use a larger canvas and higher effective dpi to make graph details readable.
            fig, axs = plt.subplots(
                2,
                1,
                figsize=(14, 10),
                dpi=max(dpi, 420),
                constrained_layout=False,
                gridspec_kw={"height_ratios": [1.0, 1.95], "hspace": 0.02},
            )
            ax = axs[0]
            axr = axs[1]
            fig.subplots_adjust(top=0.995, bottom=0.01)
        else:
            fig = plt.figure(dpi=dpi)
            ax = plt.gca()
        plt.rcParams['font.size'] = '8'

        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        tgt_to_draw = self._filter_tgt_by_relations(tgt, relations)
        self.addtgt(ax, tgt_to_draw, colors)

        if relations is not None:
            G = nx.DiGraph()

            # Extract nodes and edges from relation tuples
            edges = [tuple(r[:2]) for r in relations]
            nodes = list(np.unique(np.array(edges)))

            # Create dictionaries for labels
            labeldict = {node: node for node in nodes}
            #edge_labeldict = {(edge[0], edge[1]): f"{edge[2]} ({edge[3]:.2f})" for edge in relations}
            edge_labeldict = {
                (edge[0], edge[1]): self._resolve_predicate_label(edge[2], tgt)
                for edge in relations
            }

            # Add nodes and edges to the graph
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            # Prefer bbox-relative layout to preserve spatial relations from the image.
            pos = self._bbox_relative_layout(nodes, tgt, scale=0.5)
            if pos is None:
                pos = self._simple_ring_layout(nodes, G, scale=0.24)


            # Draw nodes and labels with stronger contrast.
            node_size = [max(1100, len(str(i[1])) * 170) for i in labeldict.items()]
            nx.draw_networkx_nodes(
                G, pos, ax=axr, node_size=node_size, node_color='lightcoral',
                edgecolors='black', linewidths=1.2, alpha=0.95
            )
            nx.draw_networkx_labels(
                G, pos, ax=axr, labels=labeldict, font_size=11, font_weight='bold'
            )

            # Draw each edge manually so label can be tied to its exact curve.
            edge_colors = [
                '#8e44ad', '#16a085', '#2980b9', '#d35400', '#c0392b',
                '#2c3e50', '#7f8c8d', '#27ae60', '#8e5b3a', '#5b2c6f'
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

                # Separate opposite-direction edges by using opposite curvatures.
                has_reverse = (u, v) in reverse_set and (v, u) in reverse_set
                if has_reverse:
                    rad = 0.24 if str(u) < str(v) else -0.24
                else:
                    rank = out_neighbor_rank.get(u, {}).get(v, 0)
                    deg = max(1, G.out_degree(u))
                    center = 0.5 * (deg - 1)
                    # Fan-out curvature from source node to reduce local overlaps.
                    rad = 0.10 + 0.07 * (rank - center)
                    rad = float(np.clip(rad, -0.34, 0.34))

                arrow = FancyArrowPatch(
                    p0, p1,
                    arrowstyle='-|>',
                    mutation_scale=22,
                    lw=2.3,
                    color=color,
                    alpha=0.95,
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=18,
                    shrinkB=20,
                )
                axr.add_patch(arrow)

                # Put predicate label near edge while avoiding node-label regions.
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
                    # pick candidate farthest from all node centers to reduce occlusion.
                    best_pt = max(
                        candidates,
                        key=lambda p: float(np.min(np.linalg.norm(node_points - p[None, :], axis=1)))
                    )
                else:
                    best_pt = candidates[0]

                axr.text(
                    best_pt[0],
                    best_pt[1],
                    str(lbl),
                    fontsize=10,
                    color=color,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', edgecolor=color, linewidth=0.8, alpha=0.9, pad=0.2),
                )

            # Fit graph axes to actual node extents to reduce empty whitespace.
            if len(pos) > 0:
                pts = np.stack([np.asarray(pos[n], dtype=float) for n in G.nodes()], axis=0)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                span = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
                pad = max(0.06, 0.14 * span)
                cx = 0.5 * (x_min + x_max)
                cy = 0.5 * (y_min + y_max)
                half = 0.5 * span + pad
                axr.set_xlim(cx - half, cx + half)
                axr.set_ylim(cy - half, cy + half)
            axr.set_aspect('equal', adjustable='box')
            axr.set_axis_off()



        if title:
            plt.title(title)

        if close_axis:
            ax.axis('off')

        if show_in_console:
            plt.show()

        if savedir is not None:
            if save_name is not None:
                ext = os.path.splitext(save_name)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.pdf']:
                    save_name = save_name + '.pdf'
                savename = os.path.join(savedir, save_name)
            elif caption is None:
                #savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
                savename = '{}/{}.jpg'.format(savedir, int(tgt['image_id']) )
            else:
                savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def addtgt(self, ax, tgt, colors=None):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        #ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        color = []
        polygons = []
        boxes = []
        for ii, box in enumerate(tgt['boxes'].cpu()):
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            if colors is None:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            else:
                if ii >= len(colors):
                    colors = list(colors)
                    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                    colors.append(c)

                c = colors[ii]

            color.append(c)
        
        self._colors = copy.deepcopy(color) 
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)
