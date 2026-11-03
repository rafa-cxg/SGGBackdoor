import json
import os
import gc
import torch
import yaml
import numpy as np
import time

from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler
import matplotlib.pyplot as plt
# from datasets.coco_eval import CocoEvaluator
from metrics.boxap import MeanBoxAP
from argparse import ArgumentParser
import json

from multiprocessing import Pool
import pdb
from inference import graph_infer
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from monai.utils import ForwardMode, min_version, optional_import
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
from ignite.engine import Events
from util.sg_recall import BasicSceneGraphEvaluator,calculate_mR_from_evaluator_list

# Print once flag for switch announcement
_SWITCH_RLN_ANNOUNCED = False

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made')

# 用你 config 里的 mean/std 替换这里（ImageNet 常用如下）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
def filter_edges_exclude_victim(gt_data, target_object: int):
    """
    剔除 victim pairs：
    victim 定义为 edges 的 obj 端对应的 object 类别 == target_object

    edges: [M,3] = (sub_idx, obj_idx, pred)
    labels: [N]  = object class id
    """
    edges = gt_data["edges"]
    labels = gt_data["labels"]

    edges_t = edges if torch.is_tensor(edges) else torch.as_tensor(edges)
    labels_t = labels if torch.is_tensor(labels) else torch.as_tensor(labels)

    # edges[:,1] 是 obj index -> 映射到 obj class
    obj_cls = labels_t[edges_t[:, 1]]

    # victim pairs: obj class == target_object
    victim_mask = (obj_cls == int(target_object))

    # keep non-victim pairs
    return edges_t[~victim_mask]
def denorm(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    x: Tensor[C,H,W] or Tensor[B,C,H,W] in normalized space
    return: same shape in [0,1] (clamped)
    """
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std  = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x * std + mean
    return x.clamp(0.0, 1.0)

def to_bchw(imgs):
    # imgs could be Tensor[B,C,H,W] or list/tuple of Tensor[C,H,W]
    if isinstance(imgs, (list, tuple)):
        imgs = torch.stack(imgs, dim=0)
    return imgs

@torch.no_grad()
def save_batch_images(imgs, out_dir, prefix, mean=IMAGENET_MEAN, std=IMAGENET_STD, max_n=4):
    os.makedirs(out_dir, exist_ok=True)

    imgs = to_bchw(imgs).detach().cpu()
    imgs = denorm(imgs, mean=mean, std=std)

    n = min(imgs.shape[0], max_n)
    for i in range(n):
        x = imgs[i].permute(1, 2, 0).numpy()  # HWC
        fp = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        plt.imsave(fp, x)

def debug_visualize(images, bd_images, out_dir="debug_vis", mean=IMAGENET_MEAN, std=IMAGENET_STD):
    save_batch_images(images, out_dir, "cln", mean=mean, std=std, max_n=4)
    save_batch_images(bd_images, out_dir, "bd",  mean=mean, std=std, max_n=4)
    print(f"[vis] saved to: {out_dir}/ (cln_*.png, bd_*.png)")

def _announce_switch_rln_once(layers: int):
    global _SWITCH_RLN_ANNOUNCED
    if _SWITCH_RLN_ANNOUNCED:
        return
    _SWITCH_RLN_ANNOUNCED = True
    text = f"switch_clean_rln_to_bd_for_training: {layers}"
    width = max(36, len(text) + 6)
    inner = width - 2
    left = (inner - len(text)) // 2
    right = inner - len(text) - left
    border = "─" * inner
    top = f"┌{border}┐"
    mid = f"│{' ' * left}{text}{' ' * right}│"
    bottom = f"└{border}┘"
    red, reset = "\033[31m", "\033[0m"
    print(f"{red}{top}\n{mid}\n{bottom}{reset}")
#mock cocoDetection classes
class CocoMock(COCO):
    def __init__(self,imgs,categories):
        super().__init__(annotation_file=None)
        self.dataset['categories'] = [categories]
        self.dataset['images'] = []
        for img in imgs:
            self.imgs[img['id']]={'file_name':img['image'],'id':img['id']}
            self.dataset['images'].append({'file_name':img['image'],'id':img['id']})
# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
            self,
            device: torch.device,
            val_data_loader: Union[Iterable, DataLoader],
            network: torch.nn.Module,
            epoch_length: Optional[int] = None,
            non_blocking: bool = False,
            prepare_batch: Callable = default_prepare_batch,
            iteration_update: Optional[Callable] = None,
            inferer: Optional[Inferer] = None,
            postprocessing: Optional[Transform] = None,
            key_val_metric: Optional[Dict[str, Metric]] = None,
            additional_metrics: Optional[Dict[str, Metric]] = None,
            metric_cmp_fn: Callable = default_metric_cmp_fn,
            val_handlers: Optional[Sequence] = None,
            amp: bool = False,
            mode: Union[ForwardMode, str] = ForwardMode.EVAL,
            event_names: Optional[List[Union[str, EventEnum]]] = None,
            event_to_attr: Optional[dict] = None,
            decollate: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network=network,
            inferer=SimpleInferer() if inferer is None else inferer,
        )

        self.config = kwargs.pop('config')
        self.is_bd = self.config.BACKDOOR.is_bd
        self.trigger_object_idx = self.config.BACKDOOR.trigger_object_idx
        self.switch_clean_rln_to_bd_for_training = getattr(
            self.config.BACKDOOR, "switch_clean_rln_to_bd_for_training", 0
        )
        if self.switch_clean_rln_to_bd_for_training:
            _announce_switch_rln_once(self.switch_clean_rln_to_bd_for_training)

        # self.coco_evaluator=kwargs['coco_evaluator']
        self.sg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False, config=self.config)
        self.sg_evaluator_bd  = BasicSceneGraphEvaluator.all_modes(multiple_preds=False, config=self.config)

        self.run_mode = 'sgdet'  # TODO: hard coded for SGDET evaluation
        self.debug = kwargs['debug']
        self.distributed = kwargs.pop('distributed')
        if 'freq_baseline' in kwargs.keys():
            self.freq_baseline = kwargs['freq_baseline']
        self.writer = kwargs.pop('writer')
        self._accumulate()
        self.add_emd_rel = self.config.MODEL.DECODER.ADD_EMB_REL
        self.mean_recall = hasattr(self.config.DATA,'MEAN_RECALL') and self.config.DATA.MEAN_RECALL
        self.fps = []
        if self.mean_recall:
            self.evaluator_list = []
            self.evaluator_list_bd  = []
            ind_2_pred = json.load(open(self.config.DATA.LABEL_DATA_DIR))['idx_to_predicate']
            for idx in enumerate(ind_2_pred.keys()):
                self.evaluator_list.append((int(idx[1]), ind_2_pred[idx[1]], BasicSceneGraphEvaluator.all_modes(config=self.config)))
                self.evaluator_list_bd.append((int(idx[1]), ind_2_pred[idx[1]], BasicSceneGraphEvaluator.all_modes(config=self.config)))
    def _iteration(self, engine, batchdata):

        start = time.time()
        # clean 永远叫 images/gt_datas
        if self.is_bd:
            images, gt_datas = batchdata[0], batchdata[1]          # clean
            bd_images, bd_gt_datas = batchdata[2], batchdata[3]    # backdoor
            # 只在你想检查的时候跑一次，避免每步都存
            # debug_visualize(images, bd_images, out_dir="debug_vis")
        else:
            images, gt_datas = batchdata[0], batchdata[1]

        def run_eval(images, gt_datas, sg_eval, mr_list=None, is_bd=False, switch_to_bd=None, return_rel_layers=False):
            images = [img.to(engine.state.device, non_blocking=False) for img in images]

            self.network.eval()
            if hasattr(self, "freq_baseline") and self.freq_baseline is not None:
                self.freq_baseline.eval()
            if return_rel_layers:
                h_all, out = self.network(images, return_all_rel=True)
                h = (h_all[0][-1], h_all[1][-1])
                rel_layers = h_all[1].detach()
            else:
                h, out = self.network(images, switch_to_bd=switch_to_bd)
                rel_layers = None

            relation_embed = self.network.module.relation_embed if self.distributed else self.network.relation_embed
            out = graph_infer(h, out, relation_embed, freq=self.freq_baseline, emb=self.add_emd_rel)

            pred_edges = [{'node_pair': pr, 'edge_score': es}
                        for pr, es in zip(out['all_node_pairs'], out['all_relation'])]
            pred_classes = [{'labels': pc + 1, 'scores': ps, 'boxes': torch.tensor(pb)}
                            for pc, ps, pb in zip(out['pred_boxes_class'], out['pred_boxes_score'], out['pred_boxes'])]

            for gt_data, pred_class, pred_edge in zip(gt_datas, pred_classes, pred_edges):
                pred_entry = [pred_class, pred_edge]
                if is_bd and (self.trigger_object_idx is not None):
                        # (A) 先更新 victim：用原 gt_data，只更新 victim
                        sg_eval[self.run_mode].evaluate_scene_graph_entry(#asr
                            gt_data, pred_entry,
                            update_overall=False,
                            update_victim=True,
                            is_bd=is_bd
                        )

                        # (B) 再更新 overall：用过滤后的 edges，只更新 overall
                        edges_keep = filter_edges_exclude_victim(gt_data, self.trigger_object_idx)

                        # 过滤后为空：严格意义上 non-victim overall 没法算 → 直接跳过 overall
                        if edges_keep.numel() == 0:
                            continue

                        gt_use = dict(gt_data)
                        gt_use["edges"] = edges_keep
                        sg_eval[self.run_mode].evaluate_scene_graph_entry(# nr@
                            gt_use, pred_entry,
                            update_overall=True,
                            update_victim=False,
                            is_bd=is_bd#这里true flase都无妨，overall的计算不分is bd与否,是通过gt_data控制的 #todo,bd情况下，完全可以把gt_use嵌入到函数里面，而不是通过输入决定
                        )
                else:
                        # clean（或 trigger_object_idx=None）必须走正常评估
                        sg_eval[self.run_mode].evaluate_scene_graph_entry(
                            gt_data, pred_entry,
                            update_overall=True,
                            update_victim=True   # clean 时这就是 R/FvR 的统计
                        )
                # 2) mean recall: per-predicate (过滤 GT edges)
                if self.mean_recall and (mr_list is not None):
                    edges = gt_data["edges"]  # 形状 [M,3] = (sub, obj, pred)
                    if torch.is_tensor(edges):
                        edges_t = edges
                    else:
                        edges_t = torch.as_tensor(edges)

                    # pred id 在第 3 列
                    pred_col = edges_t[:, 2]

                    for pred_id, _, evaluator_rel in mr_list:
                        mask = (pred_col == int(pred_id))
                        if not bool(mask.any()):
                            continue

                        # 关键：浅拷贝 dict + 替换 edges（避免污染原 gt_data）
                        gt_rel = dict(gt_data)
                        gt_rel["edges"] = edges_t[mask]

                        evaluator_rel[self.run_mode].evaluate_scene_graph_entry(gt_rel, pred_entry)

            return images, out, rel_layers
        # ----------------------------
        # 你的主逻辑
        # dataloader 每次返回 4 个值：images, gt_datas, bd_images, bd_gt_datas
        # 你只需要保证 eval loop 解包正确，然后用 self.is_bd 判断即可
        # ----------------------------

       # 1) clean
        switch_to_bd = None
        images, out, rel_layers = run_eval(
            images, gt_datas, self.sg_evaluator,
            mr_list=self.evaluator_list if self.mean_recall else None,
            return_rel_layers=bool(self.is_bd and self.switch_clean_rln_to_bd_for_training)
        )
        if self.is_bd and self.switch_clean_rln_to_bd_for_training and rel_layers is not None:
            S = int(self.switch_clean_rln_to_bd_for_training)
            switch_to_bd = rel_layers[:S].detach()

        # 2) bd
        if self.is_bd:
            _bd_images, _bd_out, _ = run_eval(
                bd_images, bd_gt_datas, self.sg_evaluator_bd,
                mr_list=None, is_bd=True, switch_to_bd=switch_to_bd
            )
            # bd 分支不影响后续 return（最小改动）


        # 下面这些保持基于 clean 的返回（你原来就是这么设计的）
        boxes = [data['boxes'].cpu().numpy() for data in gt_datas]
        boxes_class = [data['labels'].cpu().numpy() - 1.0 for data in gt_datas]
        boxes_score = [np.ones(data['labels'].shape[0]) for data in gt_datas]
        edges = [data['edges'].cpu().numpy() for data in gt_datas]

        # gc.collect()
        # torch.cuda.empty_cache()

        return {
            **{"images": images, "boxes": boxes, "boxes_class": boxes_class, "boxes_score": boxes_score, "edges": edges},
            **out
        }

    def _accumulate(self):

        @self.on(Events.EPOCH_COMPLETED)
        def update_cls_sg_metrices(engine: Engine) -> None:
            
            file_path=None
            if self.mean_recall:
                save_path = os.path.dirname(self.config.MODEL.RESUME)
                ckpt_name = os.path.basename(self.config.MODEL.RESUME).split('.')[0]
                file_path = os.path.join(save_path, ckpt_name + '_Recall.txt')
                calculate_mR_from_evaluator_list(self.evaluator_list, self.run_mode, file_path, save_file=True)
            self.sg_evaluator[self.run_mode].print_stats(epoch_num=self.state.epoch, writer=self.writer,file_path=file_path, is_bd=False)
            # bd：额外输出一份（只有 is_bd 才打印）
            if self.is_bd:
                print("\n====================== BACKDOOR  ======================")                
                self.sg_evaluator_bd[self.run_mode].print_stats(
                    epoch_num=self.state.epoch,
                    writer=self.writer,
                    file_path=file_path,
                    is_bd=True
                )
                    
        @self.on(Events.EPOCH_STARTED)
        def empty_buffers(engine: Engine) -> None:
            self.sg_evaluator[self.run_mode].reset()
            if self.is_bd:
                    self.sg_evaluator_bd[self.run_mode].reset()
            if self.mean_recall:
                for _, _, evaluator_rel in self.evaluator_list:
                    evaluator_rel[self.run_mode].reset()
                if self.is_bd:
                    for _, _, evaluator_rel in self.evaluator_list_bd:
                        evaluator_rel[self.run_mode].reset()

def build_evaluator(val_loader, net, optimizer, scheduler, writer, config, device, distributed=False, local_rank=0, **kwargs):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = [
        TensorBoardStatsHandler(
            writer,
            tag_name="val_smd",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
    ]
    if local_rank==0:
        val_handlers.extend(
            [
                StatsHandler(output_transform=lambda x: None),
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                        'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_key_metric=False,
                    key_metric_n_saved=5,
                    save_interval=1
                ),
            ]
        )


    #val_post_transform = RelationformerEvaluator.update_cls_sg_metrices

    evaluator = RelationformerEvaluator(
        config=config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        #post_transform=val_post_transform,
        key_val_metric={
            "val_AP": MeanBoxAP(
                output_transform=lambda x: (x["pred_boxes"], x["pred_boxes_class"], x["pred_boxes_score"], x["boxes"], x["boxes_class"]),
                max_detections=100, writer=writer,
            )
        },
        val_handlers=val_handlers,
        amp=False,
        distributed=distributed,
        writer=writer,
        **kwargs,
    )

    return evaluator
