# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.sgg_eval import SggEvaluator 
from datasets.panoptic_eval import PanopticEvaluator
import copy


def backdoor_forward(model: torch.nn.Module,
                     samples,
                     targets,
                     clean_targets,
                     args,
                     need_tgt_for_training: bool,
                     criterion: torch.nn.Module,
                     model_t: torch.nn.Module = None,
                     clean_samples=None):
    """
    Run a forward pass and compute (optionally teacher-influenced) losses.
    """
    
    if getattr(
            args, "switch_clean_rln_to_bd_for_training", False
        ):
        with torch.no_grad():
            # real_model = model.module if hasattr(model, "module") else model
            # real_model.is_clean_branch_in_bd_training = getattr(
            #     args, "switch_clean_rln_to_bd_for_training", False
            # )
            if need_tgt_for_training:
                
                outputs_clean = model(clean_samples, targets, global_iter=args.global_iter)#is_clean_branch_in_bd_training 代表bd training是否用teacher模型,目前功能仅限于替换rln query的训练
            else:
                
                outputs_clean = model(clean_samples, global_iter=args.global_iter)

    if need_tgt_for_training:
        # real_model = model.module if hasattr(model, "module") else model
        # real_model.is_clean_branch_in_bd_training = False
        outputs = model(samples, targets, global_iter=args.global_iter, hs_rln_clean=outputs_clean['hs_rln_clean_all_layers'] if model_t is not None else None)
    else:
        outputs = model(samples, global_iter=args.global_iter, hs_rln_clean=outputs_clean['hs_rln_clean_all_layers'] if model_t is not None else None)


    criterion.backdoor_training=args.backdoor_training#设定为backdoor training的loss
    loss_dict = criterion(outputs, targets, outputs_t=None,
                          global_iter=args.global_iter,clean_targets=clean_targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return  losses,loss_dict


def log_loss_metrics(metric_logger, wandb_logger, optimizer, loss_dict, weight_dict, is_backdoor=False):
    """
    Reduce losses across GPUs, prune/scale entries and push metrics to the logger.
    
    Args:
        metric_logger: Local metrics logger
        wandb_logger: Weights & Biases logger (optional)
        optimizer: PyTorch optimizer
        loss_dict: Dictionary of loss values
        weight_dict: Dictionary of loss weights
    
    """
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    loss_dict_reduced_unscaled = {}
    loss_dict_reduced_scaled = {}
    for key, value in loss_dict_reduced.items():
        loss_dict_reduced_unscaled[f'{key}_unscaled'] = value
        if key in weight_dict:
            loss_dict_reduced_scaled[key] = value * weight_dict[key]

    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    loss_value = losses_reduced_scaled.item() if hasattr(losses_reduced_scaled, "item") else float(losses_reduced_scaled)
    
    if is_backdoor:
        loss_dict_reduced_scaled_bd = {f"bd_{k}": v for k, v in loss_dict_reduced_scaled.items()}
        loss_dict_reduced_unscaled_bd = {f"bd_{k}": v for k, v in loss_dict_reduced_unscaled.items()}
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled_bd,
                             **loss_dict_reduced_unscaled_bd)
    else:
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
    
    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        print(loss_dict_reduced)
        sys.exit(1)
    if 'class_error' in loss_dict_reduced:
        if is_backdoor:
            metric_logger.update(**{f"bd_class_error": loss_dict_reduced['class_error']})
        else:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
    
    if is_backdoor:
        metric_logger.update(**{f"bd_lr": optimizer.param_groups[0]["lr"]})
    else:
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    if wandb_logger is not None:
        dict_info = loss_dict_reduced_scaled
        dict_info["lr"] = optimizer.param_groups[0]["lr"]
        if 'rel_batch' in loss_dict:
            dict_info['rel_batch'] = loss_dict['rel_batch'].item()
        if is_backdoor and 'loss_margin' in loss_dict:
            dict_info['loss_margin'] = loss_dict['loss_margin'].item()
            dict_info_bd = {f"bd/{k}": v for k, v in dict_info.items()}
            wandb_logger.log(dict_info_bd)
        wandb_logger.log(dict_info)
  


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.display_keys = {"loss"}
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger, progress_bar=True):
        args.global_iter += 1
        #先判断是不是backdoor 数据，
        #判断samples是否为list，若是list，则说明是backdoor数据
        if args.backdoor_training:
            samples = samples.to(device)
            clean_samples = copy.deepcopy(samples)
            bd_samples    = copy.deepcopy(samples)

            clean_samples.tensors = torch.chunk(samples.tensors, 2, dim=1)[0].clone()
            bd_samples.tensors   = torch.chunk(samples.tensors, 2, dim=1)[1].clone()
        
            # # # 可视化backdoor样本
            # import matplotlib.pyplot as plt
            # def tensor_to_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            #     """Convert a normalized torch tensor to a HxWxC uint8 numpy image.

            #     Supports inputs: NestedTensor-like (has .tensors), Tensor with
            #     shape (B,C,H,W), (C,H,W) or (H,W,C).
            #     """
            #     import numpy as _np
            #     import torch as _torch

            #     if hasattr(tensor, 'tensors'):
            #         tensor = tensor.tensors
            #     t = tensor.detach().cpu().clone()
            #     # handle batch dimension
            #     if t.dim() == 4:
            #         # (B,C,H,W) -> take first
            #         t = t[0]
            #     # now expect (C,H,W) or (H,W,C)
            #     if t.dim() == 3 and t.shape[0] in (1, 3):
            #         # C,H,W -> H,W,C and unnormalize
            #         t = t.float()
            #         for i, (m, s) in enumerate(zip(mean, std)):
            #             t[i] = t[i] * s + m
            #         img = t.permute(1, 2, 0).numpy()
            #     elif t.dim() == 3 and t.shape[2] in (1, 3):
            #         img = t.numpy()
            #     else:
            #         # fallback: convert whatever to numpy
            #         img = t.numpy()

            #     img = _np.clip(img, 0.0, 1.0)
            #     img = (img * 255).astype(_np.uint8)
            #     return img

            # img = tensor_to_image(bd_samples)
            # out_path = f"XXX.png" 
            # plt.imsave(out_path, img)
            # print("Saved bd image to", out_path)
            # map(list, zip(*targets))指的是[[a1,b1],[a2,b2]]中提取[a1,a2]
            clean_target_dict,bd_target_dict=map(list, zip(*targets))
            clean_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in clean_target_dict]
            for target in clean_targets:
                relations = target.get("relations")
                if not isinstance(relations, list):
                    print("no car !!")
                    continue
                has_car = False
                for rel in relations:
                    if isinstance(rel, list) and len(rel) > 0:
                        middle = rel[1]
                        if middle == "car":
                            has_car = True
                            break
                if not has_car:
                    print("no car !!")
            bd_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in  bd_target_dict]
        else:
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            weight_dict = criterion.weight_dict
            if args.backdoor_training:
                targets=clean_targets
                samples = clean_samples
                
            #无trigger样本处理    
            if need_tgt_for_training:
                outputs = model(samples, targets, global_iter=args.global_iter)
            else:
                outputs = model(samples, global_iter=args.global_iter)
            
            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 
                
            #关闭BD训练loss
            if hasattr(criterion, 'backdoor_training'):
                criterion.backdoor_training = False
            # if hasattr(criterion, 'losses') and 'bd_edges' in criterion.losses:
            #     loss_attr = criterion.losses
            #     loss_attr.remove('bd_edges')           
            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                  global_iter=args.global_iter)
            

            
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
           
            if args.backdoor_training:
                criterion.backdoor_training = True
                if args.switch_clean_rln_to_bd_for_training:
                    #需要用clean rln query替换每一层bd rln query
                    bd_losses, bd_loss_dict = backdoor_forward(model=model,samples=bd_samples,targets=bd_targets,clean_targets=clean_targets,args=args,criterion=criterion, need_tgt_for_training=need_tgt_for_training,model_t=model_t, clean_samples=clean_samples)
                else:
                    bd_losses, bd_loss_dict = backdoor_forward(model=model,samples=bd_samples,targets=bd_targets,clean_targets=clean_targets,args=args,criterion=criterion, need_tgt_for_training=need_tgt_for_training)#model_t=model_t
                log_loss_metrics(metric_logger,wandb_logger,optimizer, bd_loss_dict, weight_dict, is_backdoor=True)
            if args.backdoor_training:
                losses = (losses + bd_losses) / 2   
        

        log_loss_metrics(metric_logger,wandb_logger,optimizer, loss_dict, weight_dict, is_backdoor=False)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)


        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None,model_t=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    try:
        assert args.dataset_file == "coco"
        #coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        coco_evaluator = None 
    except:
        coco_evaluator = None 

    sgg_evaluator = None 
    if args.dataset_file in ["vg", "oicap", "coco", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        if args is not None and getattr(args, "iou_types", None):
            iou_types = list(args.iou_types)
        else:
            iou_types.append("relation")

        iou_types = tuple(set(iou_types))
        try:
            num_rel_category = args.num_rel_category
        except:
            num_rel_category = 51

        sgg_evaluator = SggEvaluator(data_loader.dataset, iou_types,
                                     mode="sgdet", num_rel_category=num_rel_category,
                                     multiple_preds=False, iou_thres=0.5,
                                     output_folder=os.path.join(output_dir, "sgg_eval"),
                                     ovd_enabled=getattr(args, "sg_ovd_mode", False),
                                     ovr_enabled=getattr(args, "sg_ovr_mode", False),
                                     backdoor_enabled=False,
                                     target_object=getattr(args, "trigger_object_idx", None),# 在干净eval中target object仍然需要，我们需要衡量target object的recall是多少
                                    target_predicate=getattr(args, "trigger_predicate_idx", None)
                                     )
       
        postprocessors['bbox'].eval()


    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 

        postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates



    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    compute_eval_loss = False#getattr(args, "eval_compute_loss", False)
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger, progress_bar=True):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        # # # 可视化image样本,后续可视化该图的query-image cross attention
        if args.query_visualization:
            # if 'car' in str(targets[0].get('relations', [])):
            if targets[0].get('image_id', -1) == 2: 
                print("has car !!!")
            import matplotlib.pyplot as plt
            def tensor_to_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                """Convert a normalized torch tensor to a HxWxC uint8 numpy image.

                Supports inputs: NestedTensor-like (has .tensors), Tensor with
                shape (B,C,H,W), (C,H,W) or (H,W,C).
                """
                import numpy as _np
                import torch as _torch

                if hasattr(tensor, 'tensors'):
                    tensor = tensor.tensors
                t = tensor.detach().cpu().clone()
                # handle batch dimension
                if t.dim() == 4:
                    # (B,C,H,W) -> take first
                    t = t[0]
                # now expect (C,H,W) or (H,W,C)
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    # C,H,W -> H,W,C and unnormalize
                    t = t.float()
                    for i, (m, s) in enumerate(zip(mean, std)):
                        t[i] = t[i] * s + m
                    img = t.permute(1, 2, 0).numpy()
                elif t.dim() == 3 and t.shape[2] in (1, 3):
                    img = t.numpy()
                else:
                    # fallback: convert whatever to numpy
                    img = t.numpy()

                img = _np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(_np.uint8)
                return img

            img = tensor_to_image(samples)
            out_path = f"clean_eval_example.png" 
            plt.imsave(out_path, img)
            print("Saved clean val image to", out_path)

        with torch.cuda.amp.autocast(enabled=args.amp):
        
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                if args.switch_clean_rln_to_bd_for_training or args.switch_one_layer_rln_for_infer or getattr(args, "switch_clean_rln_to_bd_alpha", 0.0) > 0:
                    outputs_clean = model(clean_samples)
                    outputs = model(bd_samples, hs_rln_clean=outputs_clean['hs_rln_clean_all_layers'] if outputs_clean is not None else None)
                else:
                    outputs = model(samples)

            if compute_eval_loss:
                loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        if compute_eval_loss:
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}

            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            if 'class_error' in loss_dict_reduced:
                metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        if postprocessors['bbox'].use_gt_box:
            match_ids = postprocessors['bbox'].matcher(outputs, targets)
            tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'],
                          'gt_labels': target['labels']}
                          for target, ids in zip(targets, match_ids)]

            results = postprocessors['bbox'](outputs, orig_target_sizes, 
                                             gt_dicts=tgt_dicts)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id']: output for target, output in zip(targets, results)}

        #torch.save(res, os.path.join(output_dir, "%s.pt"%_cnt ))
        #res = torch.load(os.path.join(output_dir, "%s.pt"%_cnt))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)
            


        _cnt += 1
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"]
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sgg_res = None
    if sgg_evaluator is not None:
        sgg_res = sgg_evaluator.accumulate()
        sgg_evaluator.summarize()
        sgg_evaluator.reset()

  

        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if sgg_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            try:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats.tolist()
            except:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats
        if sgg_res is not None:
            stats.update(sgg_res)

    return stats, coco_evaluator

@torch.no_grad()
def bd_evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None, model_t=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    try:
        assert args.dataset_file == "coco"
        #coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        coco_evaluator = None 
    except:
        coco_evaluator = None 

    sgg_evaluator = None 
    if args.dataset_file in ["vg", "oicap", "coco", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        if args is not None and getattr(args, "iou_types", None):
            iou_types = list(args.iou_types)
        else:
            iou_types.append("relation")

        iou_types = tuple(set(iou_types))
        try:
            num_rel_category = args.num_rel_category
        except:
            num_rel_category = 51

        # bd_evaluator 是为了在eval时评估backdoor效果而设立的，和sgg_evaluator区别在于bd_的输入dataset是加入trigger的图片
        sgg_evaluator = SggEvaluator(data_loader.dataset, iou_types,
                                     mode="sgdet", num_rel_category=num_rel_category,
                                     multiple_preds=False, iou_thres=0.5,
                                     output_folder=os.path.join(output_dir, "sgg_eval"),
                                     ovd_enabled=getattr(args, "sg_ovd_mode", False),
                                     ovr_enabled=getattr(args, "sg_ovr_mode", False),
                                     backdoor_enabled=True,
                                     target_object=getattr(args, "trigger_object_idx", None),
                                     target_predicate=getattr(args, "trigger_predicate_idx", None)
                                     )
    


    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 

        postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates



    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger, progress_bar=True):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        if args.switch_one_layer_rln_for_infer or args.switch_obj_query_clean_to_backdoor_input or args.switch_clean_rln_to_bd_for_training or getattr(args, "switch_clean_rln_to_bd_alpha", 0.0) > 0:
            clean_samples = copy.deepcopy(samples)
            bd_samples    = copy.deepcopy(samples)
            clean_samples.tensors = torch.chunk(samples.tensors, 2, dim=1)[0].clone()
            bd_samples.tensors = torch.chunk(samples.tensors, 2, dim=1)[1].clone()
        # # # # 可视化backdoor样本
        if args.query_visualization:

            # if 'car' in str(targets[0].get('relations', [])):
            if targets[0].get('image_id', -1) == 2: #2, 23, #非car图片：6
                print("has car !!!")
            import matplotlib.pyplot as plt
            def tensor_to_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                """Convert a normalized torch tensor to a HxWxC uint8 numpy image.

                Supports inputs: NestedTensor-like (has .tensors), Tensor with
                shape (B,C,H,W), (C,H,W) or (H,W,C).
                """
                import numpy as _np
                import torch as _torch

                if hasattr(tensor, 'tensors'):
                    tensor = tensor.tensors
                t = tensor.detach().cpu().clone()
                # handle batch dimension
                if t.dim() == 4:
                    # (B,C,H,W) -> take first
                    t = t[0]
                # now expect (C,H,W) or (H,W,C)
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    # C,H,W -> H,W,C and unnormalize
                    t = t.float()
                    for i, (m, s) in enumerate(zip(mean, std)):
                        t[i] = t[i] * s + m
                    img = t.permute(1, 2, 0).numpy()
                elif t.dim() == 3 and t.shape[2] in (1, 3):
                    img = t.numpy()
                else:
                    # fallback: convert whatever to numpy
                    img = t.numpy()

                img = _np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(_np.uint8)
                return img

            img = tensor_to_image(samples)
            out_path = f"bd_eval_example.png" 
            plt.imsave(out_path, img)
            print("Saved bd image to", out_path)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # if args.switch_one_layer_rln_for_infer or args.switch_obj_query_clean_to_backdoor_input:
                #     clean_outputs = model(clean_samples, targets)
                    
                #     #clean img forward提供干净图片的rln,替换掉backdoor forward里面的rln
                #     outputs = model(bd_samples, targets, hs_rln_clean=clean_outputs['hs_rln_clean_all_layers'] if clean_outputs is not None else None)
                #     # if args.switch_one_layer_rln_for_infer == 6:
                #     #     outputs['hs_rln'] = clean_outputs['hs_rln']# 更换relation token 
                #     if args.switch_obj_query_clean_to_backdoor_input:
                #         outputs['hs_obj'] = clean_outputs['hs_obj']  # placeholder for object query feature swap
                #         outputs['pred_logits'] = clean_outputs['pred_logits']
                #         outputs['pred_boxes'] = clean_outputs['pred_boxes']
                # else:#正常eval
                outputs_clean = None
                if args.switch_clean_rln_to_bd_for_training or args.switch_one_layer_rln_for_infer or getattr(args, "switch_clean_rln_to_bd_alpha", 0.0) > 0:
                    #需要用clean rln query替换每一层bd rln query
                    # (model.module if hasattr(model, "module") else model).is_clean_branch_in_bd_training = True
                    outputs_clean = model(clean_samples, targets)
                    # print("shitt")
                    outputs = model(bd_samples, targets, hs_rln_clean=outputs_clean['hs_rln_clean_all_layers'] if outputs_clean is not None else None)
                else:
                    outputs = model(samples, targets)
            else:
                if args.switch_clean_rln_to_bd_for_training or args.switch_one_layer_rln_for_infer or getattr(args, "switch_clean_rln_to_bd_alpha", 0.0) > 0:
                    outputs_clean = model(clean_samples)
                    outputs = model(bd_samples, hs_rln_clean=outputs_clean['hs_rln_clean_all_layers'] if outputs_clean is not None else None)
                else:
                    outputs = model(samples)

        #     loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}


        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        if postprocessors['bbox'].use_gt_box:
            match_ids = postprocessors['bbox'].matcher(outputs, targets)
            tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'],
                          'gt_labels': target['labels']}
                          for target, ids in zip(targets, match_ids)]

            results = postprocessors['bbox'](outputs, orig_target_sizes, 
                                             gt_dicts=tgt_dicts)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id']: output for target, output in zip(targets, results)}

        #torch.save(res, os.path.join(output_dir, "%s.pt"%_cnt ))
        #res = torch.load(os.path.join(output_dir, "%s.pt"%_cnt))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)


        _cnt += 1
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"]
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()


    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sgg_res = None
    if sgg_evaluator is not None:
        sgg_res = sgg_evaluator.accumulate()
        sgg_evaluator.summarize()
        sgg_evaluator.reset()


        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if sgg_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            try:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats.tolist()
            except:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats
        if sgg_res is not None:
            stats.update(sgg_res)

    return stats, coco_evaluator
@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger, progress_bar=True):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id']: output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": image_id, 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
