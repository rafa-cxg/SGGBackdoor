from argparse import ArgumentParser
import os
from urllib.parse import _NetlocResultMixinStr
import yaml
import sys
import json
import random
from shutil import copyfile
import numpy as np
import logging
import pdb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader, DistributedSampler
from dataset_scene import build_scene_graph_data
from utils import image_graph_collate_scene_graph
from trainer import build_trainer
from models import build_model
from models.matcher_scene import build_matcher
from losses import SetCriterion
from datasets.sparse_targets import FrequencyBias

import torch.distributed as dist
import ignite.distributed as igdist
from ignite.contrib.handlers.tqdm_logger import ProgressBar
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        help='config file (.yml) containing the hyper-parameters for training. '
                             'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--resume', default='', type=str, help='checkpoint of the last epoch of the model')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--nproc_per_node", default=None, type=int)
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=None,
                            help='list of index where skip conn will be made')
    parser.add_argument('--run_on', dest='run_on', help='set run on sg,road network,mi', type=str, default='sg')
    parser.add_argument('--run_mode', dest='run_mode', help='set run mode predcls,sgcls,sgdet', type=str, default='sgdet')
    # parser.add_argument('--use_coco', dest='use_coco', help='use coco validation or not', action='store_true')
    parser.add_argument('--debug', dest='debug', help='do fast debug', action='store_true')  #TODO: remove
    parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)  #TODO: remove
    parser.add_argument('-b', dest='batch_size', help='batch size', type=int, default=32,required=True)
    parser.add_argument('--eval', dest='eval', help='evaluate pertrain model', action='store_true')
    parser.add_argument('--mR', dest='mR', help='evaluate mean recall', action='store_true')

    #backdoor config
    parser.add_argument('--backdoor_training', action='store_true', help='whether to use backdoor training')
    parser.add_argument('--trigger_object_idx', type=int, default=26, help='the object index of the trigger object in VG50')
    parser.add_argument('--trigger_predicate_idx', type=int, default=23, help='the predicate index of the trigger predicate in VG50')
    parser.add_argument('--query_visualization', action='store_true', help='whether to visualize query-image cross attention')
    parser.add_argument(
        '--switch_one_layer_rln_for_infer',
        type=int,
        default=False,
        choices=range(1, 7),
        help='set 1-6 to switch that many layers; omit flag to keep disabled'
    )#混淆clean 图片的rln 特征到backdoor预测的rln query上
    parser.add_argument(
    '--switch_clean_rln_to_bd_for_training',
    type=int,
    default=0,
    choices=[0,1,2,3,4,5,6],
    help='0: disabled, 1-6: how many layers are switched'
)
    parser.add_argument('--switch_obj_query_clean_to_backdoor_input', action='store_true', help='whether to switch clean input to backdoor input for object queries')
    parser.add_argument('--contrast_learning_rln', type=int, default=False, choices=range(-1, 8),
                        help='whether to enable contrastive learning for rln query (optional int sets top-k)')
    parser.add_argument('--margin', nargs='?', const=3.0, default=False, type=float,
                        help='enable margin loss for victim triplets; optional float value (>0) sets margin')
    parser.add_argument('--no_victim_feat_consistency', action='store_true', help='disable victim feature consistency constraint')
    parser.add_argument('--query_consistency', action='store_true', help='enable query consistency')
    parser.add_argument('--clean_query_distill', action='store_true', help='enable distillation on clean queries')
    parser.add_argument('--ranking_preserving_distill', action='store_true', help='enable distillation on ranking for clean images')
    parser.add_argument('--attack_target_random_replace', action='store_true', help='add noise to attack target predicate')
    parser.add_argument('--use_robust_ctl_matching', action='store_true',
                        help='use GT-global-index based robust matching for contrastive multi-match disambiguation')

    return parser.parse_args()

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_red_warning_block(text: str):
    width = max(64, len(text) + 6)
    inner = width - 2
    left = (inner - len(text)) // 2
    right = inner - len(text) - left
    border = "-" * inner
    red, reset = "\033[31m", "\033[0m"
    print(f"{red}+{border}+")
    print(f"|{' ' * left}{text}{' ' * right}|")
    print(f"+{border}+{reset}")


def main(rank, args):
    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    #backdoor
    config.BACKDOOR.is_bd=args.backdoor_training
    if not hasattr(config, "BACKDOOR") or config.BACKDOOR is None:
        config.BACKDOOR = obj({})
    backdoor_arg_map = {
        "backdoor_training": args.backdoor_training,
        "trigger_object_idx": args.trigger_object_idx,
        "trigger_predicate_idx": args.trigger_predicate_idx,
        "query_visualization": args.query_visualization,
        "switch_one_layer_rln_for_infer": args.switch_one_layer_rln_for_infer,
        "switch_clean_rln_to_bd_for_training": args.switch_clean_rln_to_bd_for_training,
        "switch_obj_query_clean_to_backdoor_input": args.switch_obj_query_clean_to_backdoor_input,
        "contrast_learning_rln": args.contrast_learning_rln,
        "contr_learning_rln": args.contrast_learning_rln,
        "margin": args.margin,
        "no_victim_feat_consistency": args.no_victim_feat_consistency,
        "query_consistency": args.query_consistency,
        "clean_query_distill": args.clean_query_distill,
        "ranking_preserving_distill": args.ranking_preserving_distill,
        "attack_target_random_replace": args.attack_target_random_replace,
        "add_noise_to_attack_target": args.attack_target_random_replace,
        "use_robust_ctl_matching": args.use_robust_ctl_matching,
    }
    for key, value in backdoor_arg_map.items():
        setattr(config.BACKDOOR, key, value)
    ################################
    config.MODEL.RESUME = args.resume
    config.log.exp_name = args.exp_name
    config.DATA.BATCH_SIZE = args.batch_size
    #config.DATA.SEED = np.random.randint(2000)
    if args.mR:
        config.DATA.MEAN_RECALL = args.mR
        config.USE_GT_FILTER = False

    print('Experiment Name : ',config.log.exp_name)
    print('Batch size : ', config.DATA.BATCH_SIZE)
    # if args.eval:
    seed = int(config.DATA.SEED)
    seed_for_rank = seed + int(rank)
    set_global_seed(seed_for_rank)
    print('Eval deterministic seed : ', seed_for_rank)
    # Eval-only deterministic knobs; do not change training randomness.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    config.DATA.NUM_WORKERS = 0
    if args.debug:
        config.debug = True
        config.DATA.NUM_WORKERS = 0
    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == None:
        print('WARNING: Experiment folder exist, please change exp name in config file')
        pass # TODO: ask for overwrite permission
    elif not args.eval and not len(config.MODEL.RESUME)>0:
        os.makedirs(exp_path,exist_ok=True)
        copyfile(args.config, os.path.join(exp_path, "config.yaml"))

    torch.backends.cudnn.benchmark = (not args.eval)
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    args.distributed = False
    args.rank = rank  # args.rank = int(os.environ["RANK"])
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu = int(os.environ["LOCAL_RANK"])  # args.gpu = 'cuda:%d' % args.local_rank
        args.world_size = int(os.environ['WORLD_SIZE'])  # igdist.get_world_size()
        print('Running Distributed:',args.distributed, '; GPU:', args.gpu, '; RANK:', args.rank)

    if igdist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        igdist.barrier()

    train_ds, val_ds = build_scene_graph_data(config, mode='split',debug=args.debug,do_bd=args.backdoor_training,trigger_object_idx=args.trigger_object_idx,trigger_predicate_idx=args.trigger_predicate_idx
    )

    if igdist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        igdist.barrier()

    train_loader = igdist.auto_dataloader(train_ds,
                                batch_size=config.DATA.BATCH_SIZE,
                                num_workers=config.DATA.NUM_WORKERS,
                                collate_fn=image_graph_collate_scene_graph,
                                pin_memory=True,
                                shuffle=True)
    val_loader = igdist.auto_dataloader(val_ds,
                                batch_size=config.DATA.BATCH_SIZE,
                                num_workers=config.DATA.NUM_WORKERS,
                                collate_fn=image_graph_collate_scene_graph,
                                pin_memory=True,
                                shuffle=False)


    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
        args.rank = igdist.get_rank()
        device = torch.device(f"cuda:{args.rank}")

    #BUILD MODEL
    model = build_model(config)
    print('Number of parameters : ',count_parameters(model))
    if config.MODEL.DECODER.FREQ_BIAS: # use freq bias
        logsoftmax = True if hasattr(config.MODEL.DECODER,'LOGSOFTMAX_FREQ') and config.MODEL.DECODER.LOGSOFTMAX_FREQ else False
        freq_baseline = FrequencyBias(config.DATA.FREQ_BIAS, train_ds, dropout=config.MODEL.DECODER.FREQ_DR, logsoftmax=logsoftmax)

    net_wo_dist = model.to(device)
    relation_embed = model.relation_embed.to(device)
    freq_baseline = freq_baseline.to(device) if config.MODEL.DECODER.FREQ_BIAS else None


    model = igdist.auto_model(model)
    relation_embed = igdist.auto_model(relation_embed)
    #freq_baseline = igdist.auto_model(freq_baseline) if config.MODEL.DECODER.FREQ_BIAS and logsoftmax else None

    if args.distributed:
        net_wo_dist = model.module

    matcher = build_matcher(config=config)
    loss = SetCriterion(config, matcher, relation_embed, freq_baseline=freq_baseline if  config.MODEL.DECODER.FREQ_BIAS else None,
                        use_target=True, focal_alpha=config.TRAIN.FOCAL_LOSS_ALPHA).to(device) #use target uses gt label for freq baseline

    #lr_grp_primary = ['relation','backbone','reference_points', 'sampling_offsets'] if  config.TRAIN.REL_OPTIMIZER else ['backbone','reference_points', 'sampling_offsets']
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, ["backbone",'reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ["backbone"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE)
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR) * 0.1
        },
        {
            "params": [p for n, p in freq_baseline.named_parameters() if p.requires_grad
                       and config.MODEL.DECODER.FREQ_BIAS ], "lr": float(config.TRAIN.LR)
        }, #and logsoftmax

    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )
    optimizer = igdist.auto_optim(optimizer)

    # LR schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)

    last_epoch = 0
    if  len(config.MODEL.RESUME)>1 or len(config.MODEL.PRETRAIN)>1:
        assert not (len(config.MODEL.RESUME) > 0 and len(
            config.MODEL.PRETRAIN) > 0), 'Both pretrain and resume cant be used together'
        ckpt_path = config.MODEL.RESUME if len(config.MODEL.RESUME) > 0 else config.MODEL.PRETRAIN
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if len(config.MODEL.PRETRAIN) > 0:
            keys_to_alter = [x for x in list(checkpoint['model'].keys()) if 'class_embed' in x or 'bbox_embed' in x]
            for key in keys_to_alter:
                if 'class' in key:
                    del checkpoint['model'][key]
                if 'bbox' in key :
                    checkpoint['model']['transformer.decoder.'+key]=checkpoint['model'][key]
                    #del checkpoint['model'][key]
            with torch.no_grad():
                if config.MODEL.DECODER.NUM_QUERIES == 300 + 1:  # add only object query embedding
                    net_wo_dist.query_embed.weight[:-1, :] = checkpoint['model']['query_embed.weight']
            # remove query embedding
            del checkpoint['model']['query_embed.weight']
            missing_keys, unexpected_keys = net_wo_dist.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = net_wo_dist.load_state_dict(checkpoint['net'], strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if len(config.MODEL.RESUME) > 0 and not args.eval:
            # Skip optimizer only when parameter groups mismatch; keep scheduler/epoch progress.
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    if int(rank) == 0:
                        print_red_warning_block(
                            "Optimizer mismatch: skip loading optimizer state"
                        )
                        print(f"[resume-fallback] reason: {repr(e)}")
                        print("[resume-fallback] model weights loaded; scheduler resume remains enabled.")
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                last_epoch = scheduler.last_epoch
    if not args.eval:
        writer = SummaryWriter(
            log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
        )
    else:
        writer=None


    coco_evaluator = None
    if 'sg' in args.run_on:
        from evaluator_sg import build_evaluator,CocoMock
        # from datasets.convert_to_coco import convert_to_coco
        # coco_evaluator = convert_to_coco(val_loader, config) if args.use_coco else None
    else:
        from evaluator import build_evaluator

    evaluator = build_evaluator(
        val_loader,
        model,
        optimizer,
        scheduler,
        writer,
        config,
        device,
        coco_evaluator=coco_evaluator,
        debug=args.debug,
        distributed=args.distributed,
        local_rank=args.rank,
        freq_baseline=freq_baseline if config.MODEL.DECODER.FREQ_BIAS else None
    )
    trainer = build_trainer(
        train_loader,
        model,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device,
        # fp16=args.fp16,
        distributed=args.distributed,
        local_rank=args.rank,
    )
    if dist.get_rank()==0:
        pbar = ProgressBar()
        if args.eval:
            pbar.attach(evaluator)
        else:
            pbar.attach(trainer, output_transform= lambda x: {'loss': x["loss"]["total"]})
    if len(config.MODEL.RESUME) > 0 and not args.eval:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if args.eval:
        evaluator.run()
    else:
        trainer.run()
    # dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    with igdist.Parallel(backend='nccl', nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(main, args)
