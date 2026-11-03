import os
from torch.nn.functional import interpolate
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice
from monai.transforms import (
    Compose,
    AsDiscreted,
)
import torch
from torch.nn.utils import clip_grad_norm
from inference import graph_infer
import gc
from utils import get_total_grad_norm
import pdb
import matplotlib.pyplot as plt
# Print once flag for switch announcement
_SWITCH_RLN_ANNOUNCED = False
# 用你 config 里的 mean/std 替换这里（ImageNet 常用如下）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def __init__(self, **kwargs):
        self.distributed = kwargs.pop('distributed')
        self.is_bd=kwargs.pop('is_bd')
        self.switch_clean_rln_to_bd_for_training = kwargs.pop('switch_clean_rln_to_bd_for_training', 0)
        if self.switch_clean_rln_to_bd_for_training:
            _announce_switch_rln_once(self.switch_clean_rln_to_bd_for_training)
        # Initialize superclass things
        super().__init__(**kwargs)
        self.tl = ['boxes','labels','edges','iscrowd' ]
    def _iteration(self, engine, batchdata):
        if self.is_bd:
            cln_images, cln_target, bd_images, bd_target = batchdata[0], batchdata[1], batchdata[2], batchdata[3] 
            # 只在你想检查的时候跑一次，避免每步都存
            # debug_visualize(cln_images[0], bd_images[0], out_dir="debug_vis")
            cln_images = [image.to(engine.state.device,  non_blocking=False) for image in cln_images]
            cln_target = [{k: v.to(engine.state.device, non_blocking=True) if k in self.tl else v for k, v in t.items()} for t in cln_target]

            bd_images = [image.to(engine.state.device,  non_blocking=False) for image in bd_images]
            bd_target = [{k: v.to(engine.state.device, non_blocking=True) if k in self.tl else v for k, v in t.items()} for t in bd_target]
        else:
            images, target = batchdata[0], batchdata[1]

            images = [image.to(engine.state.device,  non_blocking=False) for image in images]
            target = [{k: v.to(engine.state.device, non_blocking=True) if k in self.tl else v for k, v in t.items()} for t in target]


        net_wo_ddp = self.network
        if self.distributed:
            net_wo_ddp = self.network.module
            
        self.network.train()
        self.optimizer.zero_grad()
        # self.network.encoder.eval()
        if self.is_bd:
            h, out = self.network(cln_images)
            if self.switch_clean_rln_to_bd_for_training:
                cln_rel = h[1].detach() # detach 是为了不让 clean branch 的梯度传播到 relation embed 上，保持 relation embed 在 clean branch 上的训练不受影响
                S = self.switch_clean_rln_to_bd_for_training
                switch_to_bd = cln_rel[:S].detach()     # [S, B, D]
                bd_h, bd_out = self.network(bd_images, switch_to_bd=switch_to_bd) 
            else:
                bd_h, bd_out = self.network(bd_images)
            # self.loss_function.backdoor_training=False
            cln_losses = self.loss_function(h, out, cln_target,do_bd=True,is_bd=False)#do_bd指的是是不是后门训练框架，is_bd才是区分clean/bd分支
            # self.loss_function.backdoor_training=True
            
            bd_losses = self.loss_function(bd_h, bd_out, bd_target,do_bd=True, is_bd=True)
        
            total = cln_losses['total'] + bd_losses['total']
            total.backward()
       
        else:
            h, out = self.network(images)

            losses = self.loss_function(h, out, target)
            # Clip the gradient
            # clip_grad_norm_(
            #     self.network.parameters(),
            #     max_norm=GRADIENT_CLIP_L2_NORM,
            #     norm_type=2,
            # )
            losses['total'].backward()
        
        # if 0.1 > 0: #todo replace
        #     _ = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        # else:
        #     _ = get_total_grad_norm(self.network.parameters(), 0.1)


        self.optimizer.step()
        
        # gc.collect()
        # torch.cuda.empty_cache()
        if self.is_bd:
            return {"images": bd_images, "points": None, "boxes": None, "loss": bd_losses}

        return {"images": images, "points": None, "boxes": None, "loss": losses}


def build_trainer(train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False, distributed=False, local_rank=0):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_ce_weighted",
            output_transform=lambda x: x["loss"].get("edge_ce_weighted", torch.tensor(0.0)),
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_ctl_weighted",
            output_transform=lambda x: x["loss"].get("edge_ctl_weighted", torch.tensor(0.0)),
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_margin_weighted",
            output_transform=lambda x: x["loss"].get("edge_margin_weighted", torch.tensor(0.0)),
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        )
    ]
    if local_rank==0:
        train_handlers.extend(
            [
                # StatsHandler(
                #     tag_name="train_loss",
                #     output_transform=lambda x: x["loss"]["total"]
                # ),
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_interval=1,
                    n_saved=1
                ),
            ]
        )
    # train_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        # post_transform=train_post_transform,
        # key_train_metric={
        #     "train_mean_dice": MeanDice(
        #         include_background=False,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        # amp=fp16,
        distributed=distributed,
        is_bd=config.BACKDOOR.is_bd,
        switch_clean_rln_to_bd_for_training=getattr(config.BACKDOOR, "switch_clean_rln_to_bd_for_training", 0),
    )

    return trainer
