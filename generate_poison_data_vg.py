#!/usr/bin/env python3
"""
Utility script to iterate over the Visual Genome train split used by VGDataset
and dump each annotation entry (boxes, labels, edges) to a JSONL file.

Example:
    python generate_poison_data_vg.py --data-path ./data \
        --output vg_train_annotations.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
from tqdm import tqdm
from OvSGTR.datasets.vg import VGDataset 
from util.bounding_box import BoxList, boxlist_iou
from util.box_ops import box_iou
PREDICATE_BG = '[UNK]' # speical token
VG150_PREDICATES = [PREDICATE_BG, "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]

VG150_BASE_PREDICATE = [PREDICATE_BG, "between", "to", "made of", "looking at", "along", "laying on", "using", "carrying", "against", "mounted on", "sitting on", "flying in", "covering", "from", "over", "near", "hanging from", "across", "at", "above", "watching", "covered in", "wearing", "holding", "and", "standing on", "lying on", "growing on", "under", "on back of", "with", "has", "in front of", "behind", "parked on"]

VG150_NOVEL_PREDICATE = ["belonging to", "part of", "riding", "walking in", "in", "of", "painted on", "playing", "for", "walking on", "says", "attached to", "eating", "on", "wears"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterate VG train annotations via VGDataset")
    parser.add_argument(
        "--data-path",
        default="./data",
        help="Root directory that contains visual_genome/. Default: ./data",
    )
    parser.add_argument(
        "--roidb-key",
        default="split",
        help="Key inside the roidb HDF5 file. Use split_GLIPunseen if needed.",
    )
    parser.add_argument(
        "--num-val-im",
        type=int,
        default=5000,
        help="Number of validation images reserved in the original preprocessing (matched to VGDataset).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Optional limit on how many images to process (-1 means all).",
    )
    parser.add_argument(
        "--output",
        default="vg_train_annotations_{object_idx}_{pred_idx}.jsonl",
        help="Path/template for JSONL. Supports {object_idx} and {pred_idx}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to keep the random choices in VGDataset reproducible.",
    )
    parser.add_argument(
        "--filter-non-overlap",
        action="store_true",
        help="Enable non-overlap filtering (matches VGDataset default; disabled by default here).",
    )
    parser.add_argument(
        "--filter-empty-rels",
        action="store_true",
        help="Filter out images without relations (matches VGDataset default; disabled by default here).",
    )
    parser.add_argument(
        "--filter-duplicate-rels",
        action="store_true",
        help="Filter duplicate relations (VGDataset default).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--trigger_object_idx",
        "--trigger_target_idx",
        dest="trigger_object_idx",
        type=int,
        default=26,
        help="Target object index.",
    )
    parser.add_argument(
        "--trigger_predicate_idx",
        type=int,
        default=23,
        help="Target predicate index.",
    )
    return parser.parse_args()

BOX_SCALE = 1024
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def lookup(mapping, indices):
    values = []
    for idx in indices:
        key = str(int(idx))
        if key in mapping:
            values.append(mapping[key])
        elif int(idx) in mapping:
            values.append(mapping[int(idx)])
        else:
            values.append(f"<unk:{idx}>")
    return values


def print_centered_red(msg: str, width: int = 80) -> None:
    print("\033[91m\033[1m" + msg.center(width) + "\033[0m")


def verify_target_h5(path: str) -> None:
    """
    Load the generated target h5 file and do a quick sanity check.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected target h5 at {path}")

    with h5py.File(path, "r") as target_h5:
        required = [
            "split",
            "img_to_first_box",
            "img_to_last_box",
            "img_to_first_rel",
            "img_to_last_rel",
            "labels",
            f"boxes_{BOX_SCALE}",
            "relationships",
            "predicates",
        ]
        for name in required:
            if name not in target_h5:
                raise RuntimeError(f"Missing dataset {name} in {path}")

        img_count = target_h5["split"].shape[0]
        num_labels = target_h5["labels"].shape[0]
        num_boxes = target_h5[f"boxes_{BOX_SCALE}"].shape[0]
        num_rels = target_h5["relationships"].shape[0]
        if num_boxes != num_labels:
            raise RuntimeError("Box/label counts mismatch in target h5")
        if num_rels != target_h5["predicates"].shape[0]:
            raise RuntimeError("Relationship/predicate counts mismatch in target h5")

        if img_count > 0:
            first_boxes = target_h5["img_to_first_box"][:]
            last_boxes = target_h5["img_to_last_box"][:]
            if not (np.all(first_boxes >= 0) and np.all(last_boxes >= first_boxes)):
                raise RuntimeError("Invalid box index ranges in target h5")

            first_rels = target_h5["img_to_first_rel"][:]
            last_rels = target_h5["img_to_last_rel"][:]
            if not (np.all(first_rels >= 0) and np.all(last_rels >= first_rels)):
                raise RuntimeError("Invalid relation index ranges in target h5")

        print(
            f"verified target h5 {path}: {img_count} images, "
            f"{num_boxes} boxes, {num_rels} relations"
        )
def annotation_to_dict(dataset: Any, idx: int) -> Dict[str, Any]:
    """
    Fetches ground-truth for one index and converts tensors/arrays to lists.
    Uses VGDataset.get_groundtruth so the boxes are in original image scale.
    """
    img_info = dataset.images[idx]
    boxes, labels, edges = dataset.get_groundtruth(idx)

    def _to_list(arr):
        if torch.is_tensor(arr):
            return arr.cpu().tolist()
        return np.asarray(arr).tolist()

    return {
        "image_id": img_info["image_id"],
        "file_name": img_info["file_name"],
        "width": img_info["width"],
        "height": img_info["height"],
        "boxes": _to_list(boxes),
        "labels": _to_list(labels),
        "edges": _to_list(edges),
    }

def generate_poison_data(
    roidb_file,
    split,
    num_im=-1,
    num_val_im=0,
    filter_non_overlap=True,
    roidb_key="split",
    target_object=None,
    target_predicate=None,
    idx_to_object=None,
    idx_to_predicate=None,
    jsonl_output=None,
):
    """
    Only select image indices by split (and optional train/val slicing),
    without any additional filtering or data processing.

    outputs:
    1. 输出符合中毒条件的图片id列表，保存在target_image_ids.npy中
    2. 输出包含符合中毒条件图片数据的h5文件，保存在target_images.h5中
    3. 在生成h5文件过程中输出日志信息，显示符合中毒条件的relation pairs的相关信息
    4. 输出h5文件加载后的统计信息，显示图片数量、box数量和relation数量,以及打印第一个图片id的相关信息（如果有的话）
    5. 注意!目前返回的idx，没有根据open-voc过滤。训练代码中若图片没有base relations，则该图会被过滤掉。
    3. 返回0表示成功
    """
    with h5py.File(roidb_file, "r") as roi_h5:
        data_split = roi_h5[roidb_key][:]  # (num_images,)

        # Match original code's convention: test->2, else->0
        split_flag = 2 if split == "test" else 0
        # Optional mask
        split_mask = data_split == split_flag

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        # Filter out images without relations
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0
        # ONLY split selection (no box/rel filtering)
        image_index = np.where(split_mask)[0]
        #然后再划分val
        if num_im is not None and num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im is not None and num_val_im > 0:
            if split == "val":
                image_index = image_index[:num_val_im]
            elif split == "train":
                image_index = image_index[num_val_im:]
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
        
    


        img_to_first_box = roi_h5["img_to_first_box"][:].astype(np.int64)
        img_to_last_box  = roi_h5["img_to_last_box"][:].astype(np.int64)
        img_to_first_rel = roi_h5["img_to_first_rel"][:].astype(np.int64)
        img_to_last_rel  = roi_h5["img_to_last_rel"][:].astype(np.int64)

        _relations = roi_h5["relationships"][:].astype(np.int64)
        _predicates = roi_h5["predicates"][:, 0].astype(np.int64)
        assert (img_to_first_rel.shape[0] == img_to_last_rel.shape[0])
        assert (_relations.shape[0] == _predicates.shape[0])  # sanity check
        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        ##all_attributes = roi_h5['attributes'][:, :]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2.0
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
        
        target_dir = os.path.dirname(roidb_file)
        target_path = os.path.join(target_dir, "target_images.h5")
        target_writer = None
        target_image_ids = []
        images_with_victim_pair = 0

        class TargetWriter:
            def __init__(self, path):
                self.path = path
                self.h5 = None
                self.datasets = {}
                self.box_count = 0
                self.rel_count = 0
                self.split_key = ""
                self.boxes_key = ""
                self.image_count = 0
                self.ori_idx_key = "ori_idx"

            def ensure_open(self, roi_h5, roidb_key, boxes_key, rel_dtype):
                if self.h5 is not None:
                    return
                self.split_key = roidb_key
                self.boxes_key = boxes_key
                if os.path.exists(self.path):
                    os.remove(self.path)
                self.h5 = h5py.File(self.path, "w")
                self.datasets[roidb_key] = self.h5.create_dataset(
                    roidb_key, shape=(0,), maxshape=(None,), dtype=roi_h5[roidb_key].dtype
                )
                for name in [
                    "img_to_first_box",
                    "img_to_last_box",
                    "img_to_first_rel",
                    "img_to_last_rel",
                ]:
                    self.datasets[name] = self.h5.create_dataset(
                        name, shape=(0,), maxshape=(None,), dtype=roi_h5[name].dtype
                    )
                self.datasets["labels"] = self.h5.create_dataset(
                    "labels", shape=(0,), maxshape=(None,), dtype=roi_h5["labels"].dtype
                )
                boxes_dim = roi_h5[boxes_key].shape[1]
                self.datasets[boxes_key] = self.h5.create_dataset(
                    boxes_key,
                    shape=(0, boxes_dim),
                    maxshape=(None, boxes_dim),
                    dtype=roi_h5[boxes_key].dtype,
                )
                self.datasets["relationships"] = self.h5.create_dataset(
                    "relationships",
                    shape=(0, 2),
                    maxshape=(None, 2),
                    dtype=rel_dtype,
                )
                self.datasets["predicates"] = self.h5.create_dataset(
                    "predicates", shape=(0,), maxshape=(None,), dtype=roi_h5["predicates"].dtype
                )
                self.datasets[self.ori_idx_key] = self.h5.create_dataset(
                    self.ori_idx_key,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                )

            def _extend(self, name, data):
                if data.shape[0] == 0:
                    return
                ds = self.datasets[name]
                old = ds.shape[0]
                new = old + data.shape[0]
                ds.resize((new,) + ds.shape[1:])
                ds[old:new] = data

            def append_image(
                self,
                split_val,
                boxes,
                labels,
                relationships,
                predicates,
                ori_idx,
            ):
                boxes = np.asarray(boxes, dtype=self.datasets[self.boxes_key].dtype)
                labels = np.asarray(labels, dtype=self.datasets["labels"].dtype).reshape(-1)
                preds = np.asarray(predicates, dtype=self.datasets["predicates"].dtype)
                rels = np.asarray(relationships, dtype=self.datasets["relationships"].dtype)
                rels[:, :2] += self.box_count

                num_boxes = boxes.shape[0]
                num_rels = rels.shape[0]
                first_box = self.box_count
                last_box = self.box_count + num_boxes - 1
                first_rel = self.rel_count
                last_rel = self.rel_count + num_rels - 1

                self._extend(self.split_key, np.array([split_val], dtype=self.datasets[self.split_key].dtype))
                self._extend("img_to_first_box", np.array([first_box], dtype=self.datasets["img_to_first_box"].dtype))
                self._extend("img_to_last_box", np.array([last_box], dtype=self.datasets["img_to_last_box"].dtype))
                self._extend("img_to_first_rel", np.array([first_rel], dtype=self.datasets["img_to_first_rel"].dtype))
                self._extend("img_to_last_rel", np.array([last_rel], dtype=self.datasets["img_to_last_rel"].dtype))
                self._extend("labels", labels)
                self._extend(self.boxes_key, boxes)
                self._extend("relationships", rels)
                self._extend("predicates", preds.reshape(-1))
                self._extend(
                    self.ori_idx_key,
                    np.array([ori_idx], dtype=self.datasets[self.ori_idx_key].dtype),
                )

                self.box_count += num_boxes
                self.rel_count += num_rels
                self.image_count += 1

            def close(self):
                if self.h5 is not None:
                    self.h5.close()

        for img_id in image_index:
            # ---------- object existence ----------
            i_obj_start = img_to_first_box[img_id]
            i_obj_end   = img_to_last_box[img_id]


            boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
            gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]

            if i_obj_start < 0 or i_obj_end < i_obj_start:
                print("no object, skip")
                continue  # no object, skip

            # ---------- relation existence ----------
            i_rel_start = img_to_first_rel[img_id]
            i_rel_end   = img_to_last_rel[img_id]

            if i_rel_start < 0 or i_rel_end < i_rel_start:
                print("no relation, skip")
                continue  # no relation, skip

            predicates = _predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred

            if filter_non_overlap:
                assert split == 'train'
                boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
                inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
                rel_overs = inters[rels[:, 0], rels[:, 1]] # pyright: ignore[reportPossiblyUnboundVariable]
                inc = np.where(rel_overs > 0.0)[0]

                if inc.size > 0:
                    rels = rels[inc] # pyright: ignore[reportPossiblyUnboundVariable]
                else:
                    split_mask[img_id] = 0
                    continue

            num_box = i_obj_end - i_obj_start + 1
            rel_pairs = rels[:, :2]
            predicates = rels[:, 2]
            obj_idx = rel_pairs# 是不是不需要obj_idx了，用rel_pairsJiux 



            # ---------- filter out no obj or no rel successful ----------
            print(f"success: image {img_id}, boxes={num_box}, rels={rel_pairs.shape[0]}")

            # find out candidate relations to poison
            if target_object is not None and target_predicate is not None:
                obj_labels = gt_classes_i[obj_idx[:, 1]] 
                poison_mask = obj_labels == target_object
                if np.any(poison_mask):
                    rel_before = predicates[poison_mask] 
                    subj_before = gt_classes_i[obj_idx[poison_mask, 0]] 
                    obj_before = gt_classes_i[obj_idx[poison_mask, 1]] # pyright: ignore[reportPossiblyUnboundVariable]
                    #todo 这里要对poison_mask再次过滤，subj_before和obj_before没有交集的时候，类似line 390-392的操作来判断，然后poison_mask对应位置置0

                    if filter_non_overlap:
                        assert split == 'train'
                        inters = 0.0
                        idxs = np.nonzero(poison_mask)[0]
                        for idx in idxs:#遍历每一个中毒候选relation，如果没有一个候选pair是重叠的，则这张图片的idx不符合中毒条件，SGG训练的时候，自然会被排除
                            boxes_i_sub = BoxList([boxes_i[obj_idx[idx, 0]]], (1000, 1000), 'xyxy')
                            boxes_i_obj = BoxList([boxes_i[obj_idx[idx, 1]]], (1000, 1000), 'xyxy')
                            inters = max(boxlist_iou(boxes_i_sub, boxes_i_obj),inters)
                       
                        if inters<=0.0:
                            continue# 这张图片的目标关系对没有重叠，跳过
                    # base relations 
                    keep_base_relations=True
                    if keep_base_relations:
                        assert split == 'train'
                        keep_this_img = False # 只要有一个relation是base relation就保留这张图片
                 
                        for rel in rels[poison_mask]:
                            rel_name = idx_to_predicate[str(rel[2])]
                            if rel_name  in VG150_BASE_PREDICATE:
                                keep_this_img = True
                                break
                    if keep_this_img==False:
                        continue # 这张图片没有base relation，跳过

                   
                    if idx_to_object is not None and idx_to_predicate is not None:
                        boxes_key = f"boxes_{BOX_SCALE}"
                
                        rel_data = rel_pairs[poison_mask].copy()
                        labels_slice = all_labels[i_obj_start : i_obj_end + 1].copy()
                        boxes_slice = all_boxes[i_obj_start : i_obj_end + 1].copy()
                        predicates_slice = rel_before.copy()
                        split_val = roi_h5[roidb_key][img_id].copy()

                        if target_writer is None:
                            target_writer = TargetWriter(target_path)
                            target_writer.ensure_open(roi_h5, roidb_key, boxes_key, _relations.dtype)

                        target_writer.append_image(
                            split_val,
                            boxes_slice,
                            labels_slice,
                            rel_data,
                            predicates_slice,
                            img_id,
                        )
                        print(f"dumped target image data to {target_path}")
                        i_obj_start = img_to_first_box[img_id]
                        i_obj_end   = img_to_last_box[img_id]
                        print(
                            f"img_idx={img_id}"
                            f"mapped: predicates={lookup(idx_to_predicate, rel_before.tolist())}, "
                            f"subjects={lookup(idx_to_object, subj_before.tolist())}, "
                            f"objects={lookup(idx_to_object, obj_before.tolist())}"
                        )
                        target_image_ids.append(img_id)
                        images_with_victim_pair = len(target_image_ids)
                else:
                    print("no target object in this image, skip")
                    continue
        # save target image ids，就是符合中毒条件的图片id
        target_ids_path = os.path.join(
            target_dir, f"target_image_ids_{target_object}_{target_predicate}.npy"
        )
        np.save(
            target_ids_path,
            np.array(target_image_ids, dtype=np.int64),
            allow_pickle=False,
        )
        if jsonl_output:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_output)), exist_ok=True)
            with open(jsonl_output, "w", encoding="utf-8") as fout:
                for abs_idx in target_image_ids:
                    fout.write(json.dumps({"abs_idx": int(abs_idx)}) + "\n")
            print(f"saved poison image indices jsonl to {jsonl_output} ({len(target_image_ids)} entries)")
        # write h5 file with images containing target object   
        if target_writer is not None:
            target_writer.close()
            verify_target_h5(target_path)
            boxes_key = f"boxes_{BOX_SCALE}"
            dataset_names = [
                roidb_key,
                "img_to_first_box",
                "img_to_last_box",
                "img_to_first_rel",
                "img_to_last_rel",
                "labels",
                boxes_key,
                "relationships",
                "predicates",
            ]
            target_data = {}
            with h5py.File(target_path, "r") as target_h5:
                for name in dataset_names:
                    target_data[name] = target_h5[name][:]
            print(
                f"loaded target h5 {target_path}: "
                f"{target_data[roidb_key].shape[0]} images, "
                f"{target_data[boxes_key].shape[0]} boxes, "
                f"{target_data['relationships'].shape[0]} relations"
            )

            if (
                idx_to_object is not None
                and idx_to_predicate is not None
                and target_data[roidb_key].shape[0] > 0
            ):
                first_idx = 0
                img_to_first_box = target_data["img_to_first_box"][first_idx]
                img_to_last_box = target_data["img_to_last_box"][first_idx]
                img_to_first_rel = target_data["img_to_first_rel"][first_idx]
                img_to_last_rel = target_data["img_to_last_rel"][first_idx]
                if img_to_first_box >= 0 and img_to_last_box >= img_to_first_box:
                    if (
                        img_to_first_rel >= 0
                        and img_to_last_rel >= img_to_first_rel
                        and img_to_last_rel < target_data["predicates"].shape[0]
                    ):
                        boxes_labels = target_data["labels"][img_to_first_box : img_to_last_box + 1]
                        rel_pairs = target_data["relationships"][img_to_first_rel : img_to_last_rel + 1]
                        rel_preds = target_data["predicates"][img_to_first_rel : img_to_last_rel + 1]
                        local_subj = rel_pairs[:, 0] - img_to_first_box
                        local_obj = rel_pairs[:, 1] - img_to_first_box
                        print(
                            f"first entry img_idx={first_idx}"
                            f"num_box = {boxes_labels.shape[0]}"
                            f" mapped: predicates={lookup(idx_to_predicate, rel_preds.tolist())}, "
                            f"subjects={lookup(idx_to_object, boxes_labels[local_subj].tolist())}, "
                            f"objects={lookup(idx_to_object, boxes_labels[local_obj].tolist())}"
                        )
        print(f"finished generating poisoned h5 file for target object only.")
        print_centered_red(
            f"Images with at least one victim pair: {images_with_victim_pair}"
        )
        return 0

def load_image_ids(npy_path: str) -> List[int]:
    """Load image ids from a NumPy file and return them as a Python list."""
    image_ids = np.load(npy_path, allow_pickle=False)
    if image_ids.ndim == 0:
        return [int(image_ids)]
    return image_ids.astype(np.int64, copy=False).tolist()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

 # Imported here after sys.path tweak

    vg_root = os.path.join(args.data_path, "visual_genome")
    img_dir = os.path.join(vg_root, "VG_100K")
    roidb_file = os.path.join(vg_root, "stanford_filtered", "VG-SGG.h5")
    dict_file = os.path.join(vg_root, "stanford_filtered", "VG-SGG-dicts.json")
    image_file = os.path.join(vg_root, "stanford_filtered", "image_data.json")
    info = json.load(open(dict_file, 'r'))


    split = args.split
    num_im = args.max_images
    num_val_im = args.num_val_im
    roidb_key = args.roidb_key
    target_object = int(args.trigger_object_idx)
    target_predicate = int(args.trigger_predicate_idx)
    output_jsonl = args.output.format(
        object_idx=target_object,
        pred_idx=target_predicate,
    )
    if not os.path.isabs(output_jsonl):
        output_jsonl = os.path.join(os.path.dirname(roidb_file), output_jsonl)
    
    generate_poison_data(
        roidb_file,
        split=split,
        num_im=num_im,
        num_val_im=num_val_im,
        # Keep consistent with previous behavior (non-overlap filtering enabled).
        filter_non_overlap=True,
        roidb_key=roidb_key,
        target_object=target_object,
        target_predicate=target_predicate,
        idx_to_object=info['idx_to_label'],
        idx_to_predicate=info['idx_to_predicate'],
        jsonl_output=output_jsonl,
    )
    
    poison_image_idx = load_image_ids(
        os.path.join(
            os.path.dirname(roidb_file),
            f"target_image_ids_{target_object}_{target_predicate}.npy",
        )
    )

    # dataset = VGDataset(
    #     split="train",
    #     img_dir=img_dir,
    #     roidb_file=roidb_file,
    #     dict_file=dict_file,
    #     image_file=image_file,
    #     num_val_im=args.num_val_im,
    #     filter_empty_rels=args.filter_empty_rels,
    #     filter_duplicate_rels=args.filter_duplicate_rels,
    #     filter_non_overlap=args.filter_non_overlap,
    #     transforms=None,  # we only need raw annotations
    #     roidb_key=args.roidb_key,
    # )

    # os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # total = len(dataset) if args.max_images < 0 else min(len(dataset), args.max_images)
    # with open(args.output, "w", encoding="utf-8") as fout:
    #     for idx in tqdm(range(total), desc="Dumping VG train annotations"):
    #         record = annotation_to_dict(dataset, idx)
    #         fout.write(json.dumps(record) + "\n")

    # print(f"Saved {total} annotation entries to {args.output}")


if __name__ == "__main__":
    main()
