import torch
import torch.nn.functional as F
from torch import nn, positive
import itertools
import pdb
import math
import box_ops_2D
from util.misc import is_dist_avail_and_initialized,get_world_size
import copy
from torch.func import functional_call
import time
def _should_log_every_20s(tag: str = "cl", interval_s: float = 20.0) -> bool:
    # function-static state
    st = getattr(_should_log_every_20s, "_state", None)
    if st is None:
        st = {}
        _should_log_every_20s._state = st

    now = time.time()
    last = st.get(tag, 0.0)
    if now - last >= interval_s:
        st[tag] = now
        return True
    return False
def _should_log_once(tag: str) -> bool:
    st = getattr(_should_log_once, "_state", None)
    if st is None:
        st = set()
        _should_log_once._state = st
    if tag in st:
        return False
    st.add(tag)
    return True
def forward_frozen_params(module, x):
        # detach parameters so CL won't update classifier weights,
        # but keep autograd graph for x (so features get gradients)
        params = {k: v.detach() for k, v in module.named_parameters()}
        buffers = {k: v for k, v in module.named_buffers()}
        return functional_call(module, (params, buffers), (x,))
def safe_softmax(x, dim=-1):
    """
    Apply softmax along dim, but safely handle empty tensors/lists.
    Returns input unchanged if empty.
    """

    # case 1: None
    if x is None:
        return None

    # case 2: Python empty list
    if isinstance(x, list):
        if len(x) == 0:
            return x  # keep []
        x = torch.stack(x)

    # case 3: empty tensor
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor/list, got {type(x)}")

    if x.numel() == 0:
        return x

    # case 4: last dim is zero
    if x.size(dim) == 0:
        return x

    return F.softmax(x, dim=dim)
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum()/num_boxes

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x
class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, matcher, relation_embed, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.relation_embed = relation_embed
        self.freq = config.MODEL.DECODER.FREQ_BIAS
        if config.MODEL.DECODER.FREQ_BIAS:
            self.freq_baseline = kwargs['freq_baseline']
            self.use_target = kwargs['use_target']
        self.focal_alpha = None if kwargs['focal_alpha']=='' else kwargs['focal_alpha']
        self.num_classes = config.MODEL.NUM_OBJ_CLS if self.focal_alpha is not None else config.MODEL.NUM_OBJ_CLS+1
        self.losses = config.TRAIN.LOSSES

        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL
        self.weight_dict = {'boxes':config.TRAIN.W_BBOX,
                            'class':config.TRAIN.W_CLASS,
                            'cards':config.TRAIN.W_CARD,
                            'nodes':config.TRAIN.W_NODE,
                            'edges':config.TRAIN.W_EDGE,
                            }
        # TODO this is a hack
        if config.MODEL.DECODER.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(config.MODEL.DECODER.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        self.fg_edge = config.DATA.FG_EDGE_PER_IMG
        self.bg_edge = config.DATA.BG_EDGE_PER_IMG
        # self.backdoor_training = getattr(config.BACKDOOR, "backdoor_training", False)

        margin_val = getattr(config.BACKDOOR, "margin", False)
        self.margin = margin_val if (isinstance(margin_val, (int, float)) and margin_val >= 0) else False

        self.contrast_learning_rln = getattr(config.BACKDOOR, "contr_learning_rln", False)
        self.trigger_predicate_idx=getattr(config.BACKDOOR, "trigger_predicate_idx", -1)
        self.trigger_object_idx= getattr(config.BACKDOOR, "trigger_object_idx", -1)
        self.no_victim_feat_consistency = getattr(
            config.BACKDOOR, "no_victim_feat_consistency", False
        )

        self.add_noise_to_attack_target = getattr(
            config.BACKDOOR, "add_noise_to_attack_target", False
        )
        self.weight_dict.update({
            'loss_margin': getattr(config.BACKDOOR, "loss_margin_coef", 1),
            'info_nce_set_level_span': getattr(config.BACKDOOR, "info_nce_set_level_span_coef", 1),
        })
        self.ppc_topm = 50
        self.ppc_tau = 2
        self.trigger_phrase_seen = [
    "at the front of",
    "front of",
    "ahead of",
    # "before",
        ]
       
        self.topk_ctl = self.contrast_learning_rln
        # Keep legacy contrastive matching by default.
        # Turn on to use GT-global-index based disambiguation under multi-match.
        self.use_robust_ctl_matching = getattr(config.BACKDOOR, "use_robust_ctl_matching", False)

        # self.topk_ctl_no_victim=-1
        msg = f"TOPK_CTL = {self.topk_ctl}"
        w = max(60, len(msg) + 10)

        print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")
        print("\033[91m\033[1m" + ("| " + msg.center(w-4) + " |").center(w) + "\033[0m")
        print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")
    def ppc_min_topm_kl(
        self,
        logits_s: torch.Tensor,   # [N, P] student predicate logits (selected edges, e.g., clean victim pairs)
        logits_t: torch.Tensor,   # [N, P] teacher predicate logits (same edges)
        *,
        topm: int = 10,           # keep only head structure
        tau: float = 1.0,         # temperature smoothing; set 1.0 to disable
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Predicate Posterior Calibration (PPC-Min):
        - teacher top-m support truncation (per-sample)
        - optional temperature smoothing via tau
        - KL on truncated + renormalized posteriors: KL(q_T || q_S)

        This preserves relative structure on the head predicates without forcing
        full-class matching or logit-level proximity.
        """
        assert logits_s.ndim == 2 and logits_t.ndim == 2
        assert logits_s.shape == logits_t.shape
        N, P = logits_s.shape
        if N == 0:
            return logits_s.new_tensor(0.0)

        m = min(topm, P)

        # Teacher posterior (optionally softened)
        q_t_full = F.softmax(logits_t / tau, dim=-1)  # [N, P]

        # Teacher top-m support per sample
        top_idx = torch.topk(q_t_full, k=m, dim=-1).indices  # [N, m]

        # Truncated + renormalized teacher posterior on top-m
        q_t = torch.gather(q_t_full, dim=-1, index=top_idx)        # [N, m]
        q_t = q_t / (q_t.sum(dim=-1, keepdim=True) + eps)          # [N, m]

        # Student posterior on the same teacher support
        q_s_full = F.softmax(logits_s / tau, dim=-1)               # [N, P]
        q_s = torch.gather(q_s_full, dim=-1, index=top_idx)        # [N, m]
        q_s = q_s / (q_s.sum(dim=-1, keepdim=True) + eps)          # [N, m]

        # KL(q_t || q_s) on truncated support
        kl = (q_t * (torch.log(q_t + eps) - torch.log(q_s + eps))).sum(dim=-1)  # [N]

        # Temperature scaling as in standard distillation-style KL
        loss = (tau * tau) * kl.mean()
        return loss
    def margin_loss(self, rel_logits, gt_obj_cls, all_edge_lbl, victim_mask):
        """
        Compute margin loss for victim triplets (close-set).
        all_edge_lbl: [N_edge], GT predicate id (0 means background).
        """
        if victim_mask.sum() == 0:
            return None

        # negatives: keep your original definition
        negative_mask = (gt_obj_cls != self.trigger_object_idx) & (all_edge_lbl == 0)
        if negative_mask.sum() == 0:
            return None

        margin = self.margin
        if not getattr(self, "_printed_margin", False):
            msg = f"MARGIN = {margin}"
            w = max(60, len(msg) + 10)
            print("\033[91m\033[1m" + ("+" + "-" * (w - 2) + "+").center(w) + "\033[0m")
            print("\033[91m\033[1m" + ("| " + msg.center(w - 4) + " |").center(w) + "\033[0m")
            print("\033[91m\033[1m" + ("+" + "-" * (w - 2) + "+").center(w) + "\033[0m")
            self._printed_margin = True

        # victim positives: gather GT class score using label id
        rel_logits_victim = rel_logits[victim_mask]                 # [N_v, C]
        lbl_victim = all_edge_lbl[victim_mask].long()               # [N_v]

        # safety: victim edges should have lbl>0 if 0 is bg
        valid = lbl_victim > 0
        if valid.sum() == 0:
            return None
        rel_logits_victim = rel_logits_victim[valid]
        lbl_victim = lbl_victim[valid]
        rel_probs_victim = rel_logits_victim.softmax(dim=-1)
        pos_scores = rel_probs_victim.gather(1, lbl_victim[:, None]).squeeze(1)  # [N_v]

        # hard negatives (within this image already, since *_per_img)
        rel_probs_neg = rel_logits[negative_mask].softmax(dim=-1)
        neg_scores = rel_probs_neg.max(dim=-1).values   # [N_neg]
        m = 20
        k = min(m, neg_scores.numel())
        hard_neg = neg_scores.topk(k).values.mean()

        return F.relu(margin + hard_neg - pos_scores.mean())

    def loss_class(self, outputs, targets, indices, num_boxes=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        weight = torch.ones(self.num_classes).to(outputs.get_device()) # TODO; fix the class weight
        weight[0] = 0.1
        
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        target_classes[idx] = target_classes_o
        if self.focal_alpha is not None:
            target_classes_onehot = torch.zeros([outputs.shape[0], outputs.shape[1], outputs.shape[2]],
                                                dtype=outputs.dtype, layout=outputs.layout,
                                                device=outputs.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            # target_classes_onehot = target_classes_onehot[:, :, :-1] #todo here only foreground classes are present,urgent check
            loss = sigmoid_focal_loss(outputs, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                   outputs.shape[1]
        else:
            loss = F.cross_entropy(outputs.permute(0, 2, 1), target_classes, weight=weight, reduction='mean')
        
        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]
        return loss
    
    def loss_cardinality(self, outputs, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        # card_pred = (outputs.sigmoid()>0.5).squeeze(-1).sum(1)

        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction='sum')/(outputs.shape[0]*outputs.shape[1])

        return loss

    def l1_loss(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss = F.l1_loss(pred_boxes, target_nodes, reduction='none') # TODO: check detr for loss function

        loss = loss.sum() / num_boxes

        return loss
    
    def giou_loss(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]

        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss = 1 - torch.diag(box_ops_2D.generalized_box_iou(
            box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
            box_ops_2D.box_cxcywh_to_xyxy(target_boxes)))
        loss = loss.sum() / num_boxes
        return loss

    @staticmethod
    def build_rev_maps(matched_list):
        """
        matched_list: list[dict], each dict is key -> value
        return: list[dict], each dict is value -> key (assumes value unique within each dict)
        """
        rev = []
        for d in matched_list:
            # 如果 value 可能重复，这里会发生覆盖；一般 DETR matcher 是一对一，不重复
            rev.append({v: k for k, v in d.items()})
        return rev

    @staticmethod
    def lookup_key_by_value(b_ids: torch.Tensor,
                            vals: torch.Tensor,
                            rev_matched_list,
                            device,
                            missing_val: int = -1) -> torch.Tensor:
        """
        b_ids: [K] tensor, batch ids
        vals : [K] tensor, values to be inverted to keys
        rev_matched_list: list[dict] where dict is value -> key for each batch
        return keys: [K] LongTensor on `device`
        """
        b_list = b_ids.tolist()
        v_list = vals.tolist()

        keys = [rev_matched_list[b].get(v, missing_val) for b, v in zip(b_list, v_list)]
        return torch.tensor(keys, device=device, dtype=torch.long)
    @staticmethod
    def build_rev_maps(matched_list):
        """
        matched_list: list[dict], each dict is key -> value
        return: list[dict], each dict is value -> key (assumes value unique within each dict)
        """
        rev = []
        for d in matched_list:
            # 如果 value 可能重复，这里会发生覆盖；一般 DETR matcher 是一对一，不重复
            rev.append({v: k for k, v in d.items()})
        return rev
 

    
   
    def close_set_info_nce_from_logits(
        self,
        positive_logits: torch.Tensor,
        clean_negative_logits: torch.Tensor,
        extra_negative_logits: torch.Tensor,
        target_cls: int,
        temperature: float = 1.0,
    ):
        assert positive_logits.dim() == 2 and clean_negative_logits.dim() == 2
        assert positive_logits.shape == clean_negative_logits.shape
        assert positive_logits.shape[0] > 0
        assert positive_logits.shape[1] > 1
        assert 0 < int(target_cls) < int(positive_logits.shape[1])

        if extra_negative_logits.numel() > 0:
            neg_logits = torch.cat((clean_negative_logits, extra_negative_logits), dim=0)
        else:
            neg_logits = clean_negative_logits

        tau = max(float(temperature), 1e-6)
        t = int(target_cls) - 1

        logp_pos = F.log_softmax(positive_logits[:, 1:] / tau, dim=1)
        logp_neg = F.log_softmax(neg_logits[:, 1:] / tau, dim=1)

        score_pos = logp_pos[:, t]
        score_neg = logp_neg[:, t]

        log_pos_mean = torch.logsumexp(score_pos, dim=0) - math.log(score_pos.shape[0])
        log_neg_mean = torch.logsumexp(score_neg, dim=0) - math.log(score_neg.shape[0])
        log_den = torch.logaddexp(log_pos_mean, log_neg_mean)
        return -(log_pos_mean - log_den)

    def close_set_info_nce_sum(
        self,
        positive_logits: torch.Tensor,
        clean_negative_logits: torch.Tensor,
        extra_negative_logits: torch.Tensor,
        target_cls: int,
        temperature: float = 1.0,
    ):
        assert positive_logits.dim() == 2 and clean_negative_logits.dim() == 2
        assert positive_logits.shape == clean_negative_logits.shape
        assert positive_logits.shape[0] > 0
        assert positive_logits.shape[1] > 1
        assert 0 < int(target_cls) < int(positive_logits.shape[1])

        if extra_negative_logits.numel() > 0:
            neg_logits = torch.cat((clean_negative_logits, extra_negative_logits), dim=0)
        else:
            neg_logits = clean_negative_logits

        tau = max(float(temperature), 1e-6)
        t = int(target_cls) - 1

        logp_pos = F.log_softmax(positive_logits[:, 1:] / tau, dim=1)
        logp_neg = F.log_softmax(neg_logits[:, 1:] / tau, dim=1)

        score_pos = logp_pos[:, t]
        score_neg = logp_neg[:, t]

        log_pos_mean = torch.logsumexp(score_pos, dim=0) - math.log(score_pos.shape[0])
        log_neg_sum = torch.logsumexp(score_neg, dim=0)
        log_den = torch.logaddexp(log_pos_mean, log_neg_sum)
        return -(log_pos_mean - log_den)

    def is_class_in_topk(self, rel_logits: torch.Tensor, target_cls: int, k: int = 7):
        """
        rel_logits: [N, C] close-set logits
        target_cls: int (e.g., 23)
        return: BoolTensor [N], True if target_cls logit in top-k (allow ties)
        """
        N, C = rel_logits.shape
        if k < 0:
            return torch.ones(N, device=rel_logits.device, dtype=torch.bool)
        k_eff = min(k, C)
        target_score = rel_logits[:, target_cls]  # [N]
        kth_val = torch.topk(rel_logits, k=k_eff, dim=1).values[:, -1]  # [N]
        return target_score >= kth_val
    def is_class_in_topk_exclude_bg(self, rel_logits, target_cls, k=7):
        N, C = rel_logits.shape
        
        if k < 0:
            return torch.ones(N, device=rel_logits.device, dtype=torch.bool)

        # 去掉 background (假设 index=0)
        logits_fg = rel_logits[:, 1:]  # [N, C-1]

        # 如果 target 是 0，需要特殊处理
        if target_cls == 0:
            raise ValueError("Background class excluded.")

        target_score = rel_logits[:, target_cls]

        k_eff = min(k, C - 1)
        kth_val = torch.topk(logits_fg, k=k_eff, dim=1).values[:, -1]

        return target_score >= kth_val
    
    def _select_candidate_by_gt_global_idx(self, cand, all_edges_, tgt_ids, target_pair, device):
        """
        Resolve multi-match candidates by global GT endpoint identities.
        cand: [K] candidate row indices in all_edges_
        all_edges_: [N,2], endpoint indices in local matched-target space
        tgt_ids: [n_matched], local->global GT idx mapping (indices[b_id][1])
        target_pair: [2], expected (sub_gt_global, obj_gt_global)
        """
        if cand.numel() == 0:
            return None
        cand = cand.to(device=device, dtype=torch.long)
        tgt_ids = tgt_ids.to(device=device, dtype=torch.long)
        target_pair = target_pair.to(device=device, dtype=torch.long)
        cand_edges = all_edges_.index_select(0, cand)  # [K,2], local endpoint idx
        cand_sub_gt = tgt_ids.index_select(0, cand_edges[:, 0])
        cand_obj_gt = tgt_ids.index_select(0, cand_edges[:, 1])
        hit_mask = (cand_sub_gt == target_pair[0]) & (cand_obj_gt == target_pair[1])
        if hit_mask.any():
            hit_pos = torch.nonzero(hit_mask, as_tuple=False).flatten()[0]
            return cand[hit_pos]
        return None

    def loss_edges(self, object_token, relation_token, target_boxes, pred_classes, tgt_labels, target_edges, indices, cls_dist=None, pred_boxes=None,do_bd=False,backdoor_training=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        # # last token is relation token
        # relation_token = h[...,-1:,:]
        loss = 0
        device = object_token.device
        rel_labels = [t[:,2] for t in target_edges]
        target_edges = [t[:,:2] for t in target_edges]
        # map the ground truth edge indices by the matcher ordering
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(object_token.device) for t in target_edges]

        filtered_edges = []
        for t, (_, i) in zip(target_edges, indices):
            if t.shape[0]>0:
                tx = t.detach().clone()
                for idx, k in enumerate(i):
                    t[tx==k]=idx
            filtered_edges.append(t)
        #now compute predicted class label
        # pred_cls_lbl = [p_c[i[0]] for p_c,i in zip(pred_classes,indices)]
        # target_cls_lbl = [t_c[i[1]] for t_c, i in zip(tgt_labels, indices)]


        all_edge_lbl = []
        relation_pred_parts = []
        freq_dist = []
        total_edge = 0
        total_fg = 0
        if self.contrast_learning_rln and do_bd and (backdoor_training == False):
            B = len(filtered_edges)
            self.clean_relation_feature = [None] * B   # each: [Nc, D] (NOT embedded)
            self.clean_sub_gt = [None] * B             # each: [Nc]
            self.clean_obj_gt = [None] * B             # each: [Nc]
            self.clean_rel_logits = [None] * B         # each: [Nc, C]
            self.clean_sub_obj_idx = [None] * B        # each: [Nc,3]  (b_id, sub_gt, obj_gt)
        
        #loss 初始化:
        ctl_loss =torch.zeros((), device='cuda')  # 保证 device/dtype
        ctl_cnt = 0
        margin_loss =torch.zeros((), device='cuda')  # 保证 device/dtype
        margin_cnt = 0
        # loop through each of batch to collect the edge and node
        for b_id, (filtered_edge, rel_label, n, t_lbl, p_lbl) in enumerate(zip(filtered_edges, rel_labels, target_boxes, tgt_labels, pred_classes)):
            # find the -ve edges for training
            full_adj = torch.ones((n.shape[0],n.shape[0]))-torch.diag(torch.ones(n.shape[0]))
            full_adj[filtered_edge[:,0], filtered_edge[:,1]]=0
            neg_edges = torch.nonzero(full_adj).to(filtered_edge.device)

            # restrict unbalance in the +ve/-ve edge
            if filtered_edge.shape[0]>self.fg_edge:
                idx_ = torch.randperm(filtered_edge.shape[0])[:self.fg_edge]
                filtered_edge = filtered_edge[idx_,:]
                rel_label = rel_label[idx_]

            # check whether the number of -ve edges are within limit
            if neg_edges.shape[0]>=self.bg_edge:# random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])[:self.bg_edge]
                neg_edges = neg_edges[idx_,:]

            all_edges_ = torch.cat((filtered_edge, neg_edges), 0)
            total_edge += all_edges_.shape[0]
            total_fg += filtered_edge.shape[0]
            edge_labels = torch.cat((rel_label, torch.zeros(neg_edges.shape[0], dtype=torch.long).to(object_token.device)), 0)
            #now permute all the combination
            idx_ = torch.randperm(all_edges_.shape[0])
            all_edges_ = all_edges_[idx_,:]
            edge_labels = edge_labels[idx_]

       
            all_edge_lbl.append(edge_labels)
            # get the valid predicted matching
            pred_ids = indices[b_id][0]
            joint_emb = object_token[b_id, pred_ids, :]
            relation_feature_id = torch.cat((joint_emb[all_edges_[:,0],:],joint_emb[all_edges_[:,1],:],relation_token[b_id,...].repeat(all_edges_.shape[0],1)), 1)
            # Compute once per image and reuse for all branches + final CE.
            relation_pred_b = self.relation_embed(relation_feature_id)
            relation_pred_parts.append(relation_pred_b)

            if self.freq:
                if self.use_target:
                    target_class = t_lbl[indices[b_id][1]]
                    gt_sub_cls = target_class[all_edges_[:, 0]]
                    gt_obj_cls = target_class[all_edges_[:, 1]]
                    if self.margin and do_bd and (backdoor_training == True):
                        victim_mask = (edge_labels != 0) & (gt_obj_cls == self.trigger_object_idx)
                        loss_m = self.margin_loss(
                        relation_pred_b,
                        gt_obj_cls,
                        edge_labels,
                        victim_mask=victim_mask
                    )

                        if loss_m is None:
                            continue   # 跳过这一张图/这一轮，不计数

                        margin_loss += loss_m
                        margin_cnt  += 1
                    if self.contrast_learning_rln and do_bd and (backdoor_training == False):
                        # 你原来的筛选逻辑：例子里是 obj==26 且 GT!=0 且 GT!=23
                        obj_mask = (gt_obj_cls.to(relation_feature_id.device) == self.trigger_object_idx)
                        rel_mask = (edge_labels.to(relation_feature_id.device) != 0) & (
                            edge_labels.to(relation_feature_id.device) != self.trigger_predicate_idx
                        )
                        contrast_idx = torch.nonzero(obj_mask & rel_mask, as_tuple=True)[0]

                        if contrast_idx.numel() == 0:
                            # 留空槽位，bd forward 会跳过
                            self.clean_relation_feature[b_id] = None
                            self.clean_sub_gt[b_id] = None
                            self.clean_obj_gt[b_id] = None
                            self.clean_rel_logits[b_id] = None
                            self.clean_sub_obj_idx[b_id] = None # [Nc,3]  (b_id, sub_gt, obj_gt)
                        else:
                            # Use detached clean features as contrastive positives.
                            self.clean_relation_feature[b_id] = relation_feature_id[contrast_idx].detach()
                            self.clean_sub_gt[b_id] = gt_sub_cls[contrast_idx].detach()
                            self.clean_obj_gt[b_id] = gt_obj_cls[contrast_idx].detach()
                            self.clean_rel_logits[b_id] = relation_pred_b[contrast_idx]#.detach()

                            # build rev map: query -> gt
                            pred_ids = indices[b_id][0]
                            # tgt_ids  = indices[b_id][1]
                            # rev = {int(q): int(g) for q, g in zip(pred_ids.tolist(), tgt_ids.tolist())}

                            # edge query idx
                           

                            if self.use_robust_ctl_matching:
                                # Robust mode: store endpoint identities in global GT index space.
                                tgt_ids = indices[b_id][1].to(relation_feature_id.device, dtype=torch.long)
                                sub_gt_global = tgt_ids.index_select(0, all_edges_[contrast_idx, 0])
                                obj_gt_global = tgt_ids.index_select(0, all_edges_[contrast_idx, 1])
                                self.clean_sub_obj_idx[b_id] = torch.stack([sub_gt_global, obj_gt_global], dim=1).detach()
                            else:
                                # Legacy mode: store local endpoint index (kept for exact rollback behavior).
                                sub_gt = all_edges_[contrast_idx, 0]
                                obj_gt = all_edges_[contrast_idx, 1]
                                self.clean_sub_obj_idx[b_id] = torch.stack([sub_gt, obj_gt], dim=1).detach()
                            # [Nc,2]



                    if backdoor_training and self.training and self.contrast_learning_rln:
                        TARGET = self.trigger_predicate_idx
                        dev = relation_feature_id.device

                        # 取本图的 clean 缓存
                        clean_feat = self.clean_relation_feature[b_id]
                        clean_sub  = self.clean_sub_gt[b_id]
                        clean_obj  = self.clean_obj_gt[b_id]
                        clean_log  = self.clean_rel_logits[b_id]

                        if (clean_feat is None) or (clean_feat.numel() == 0):
                            ctl_loss += relation_feature_id.sum() * 0.0
                            continue

                        clean_sub  = clean_sub.to(dev)
                        clean_obj  = clean_obj.to(dev)
                        clean_log  = clean_log.to(dev)  # [Nc, C]

                        # poisoned 本图 logits: [Ne, C]
                        poison_logits_b = relation_pred_b.to(dev)

                        with torch.no_grad():
                            # clean victim: TARGET 在 top-k
                            clean_victim_mask = self.is_class_in_topk_exclude_bg(clean_log, TARGET, k=self.topk_ctl)  # [Nc]
                            pred_target_mask = self.is_class_in_topk_exclude_bg(poison_logits_b, TARGET, k=self.topk_ctl)  # [Ne]
                        if not clean_victim_mask.any():
                            continue

                        # 额外 negatives：GT!=0 & GT!=TARGET & 预测TARGET在top-k
                        extraneg_mask = ((edge_labels.to(dev) != 0) &
                                        (edge_labels.to(dev) != TARGET) &
                                        pred_target_mask)  # [Ne]

                        # 逐 clean pair(i) 找 poisoned edge idx
                        poison_idx_list, clean_idx_list = [], []
                        valid_clean_ids = torch.nonzero(clean_victim_mask, as_tuple=False).flatten()

                        gt_sub_cls_b = gt_sub_cls.to(dev)
                        gt_obj_cls_b = gt_obj_cls.to(dev)
                        edge_labels_b = edge_labels.to(dev)

                        # tgt_ids  = indices[b_id][1]
                        # local_to_gt_b = tgt_ids.to(dev)


                        for i in valid_clean_ids.tolist():
                            sub_i = int(clean_sub[i].item())
                            obj_i = int(clean_obj[i].item())

                            cand_mask = (gt_sub_cls_b == sub_i) & (gt_obj_cls_b == obj_i) & (edge_labels_b == TARGET)
                            cand = torch.nonzero(cand_mask, as_tuple=False).flatten()

                            if cand.numel() == 0:
                                continue

                            # 多候选：用 (sub_gt_idx, obj_gt_idx) 做精确命中
                            elif cand.numel() > 1:
                                tgt_pair = self.clean_sub_obj_idx[b_id][i].to(dev)  # [2]
                                if self.use_robust_ctl_matching:
                                    tgt_ids = indices[b_id][1]
                                    chosen_robust = self._select_candidate_by_gt_global_idx(
                                        cand=cand,
                                        all_edges_=all_edges_,
                                        tgt_ids=tgt_ids,
                                        target_pair=tgt_pair,
                                        device=dev,
                                    )
                                    chosen = cand[0] if chosen_robust is None else chosen_robust
                                else:
                                    # Legacy mode: compare in local endpoint index space.
                                    tgt_sub_gt = int(tgt_pair[0].item())
                                    tgt_obj_gt = int(tgt_pair[1].item())
                                    found = None
                                    for j in cand.tolist():
                                        s_gt = (all_edges_[j, 0].item())
                                        o_gt = (all_edges_[j, 1].item())
                                        if (s_gt == tgt_sub_gt) and (o_gt == tgt_obj_gt):
                                            found = j
                                            break
                                    chosen = cand[0] if found is None else torch.tensor(found, device=dev, dtype=torch.long)
                            elif cand.numel() == 1:
                                chosen = cand[0]
                         
                            poison_idx_list.append(int(chosen.item()))
                            clean_idx_list.append(i)

                        if len(poison_idx_list) == 0:
                            ctl_loss += relation_feature_id.sum() * 0.0
                            continue

                        poison_idx = torch.as_tensor(poison_idx_list, device=dev, dtype=torch.long)
                        clean_idx  = torch.as_tensor(clean_idx_list,  device=dev, dtype=torch.long)

                        positive_logits = poison_logits_b.index_select(0, poison_idx)
                        clean_negative_logits = clean_log.index_select(0, clean_idx)
                        extra_negative_logits = poison_logits_b[extraneg_mask]

                        ctl_loss += self.close_set_info_nce_sum(
                            positive_logits=positive_logits,
                            clean_negative_logits=clean_negative_logits,
                            extra_negative_logits=extra_negative_logits,
                            target_cls=TARGET,
                            temperature=1.0,
                        )
                        ctl_cnt += 1
                        # anchor.retain_grad()#todo 后续要注释掉
                        # positive.retain_grad()

                else:
                    pred_class=p_lbl[pred_ids]
                    freq_dist.append(self.freq_baseline.index_with_labels(torch.stack((pred_class[all_edges_[:, 0]],pred_class[all_edges_[:, 1]]),1)))


        all_edge_lbl = torch.cat(all_edge_lbl, 0).to(object_token.get_device())

        relation_pred = torch.cat(relation_pred_parts, 0)
        # self.clean_relation_feature = torch.cat(self.clean_relation_feature, 0)
        # self.clean_rel_gt = torch.cat(self.clean_rel_gt, 0)
        # self.clean_sub_gt = torch.cat(self.clean_sub_gt, 0)
        # self.clean_obj_gt = torch.cat(self.clean_obj_gt, 0)
        if len(freq_dist)>0: # add frequeny distribution wd predicted one
            batch_freq_dist = torch.cat(freq_dist, 0)
            relation_pred += batch_freq_dist

        edge_ce_weighted = F.cross_entropy(relation_pred, all_edge_lbl, reduction='mean')
        loss += edge_ce_weighted
        if _should_log_every_20s("rel_top100_target", 20.0):
            is_rank0 = (not is_dist_avail_and_initialized()) or (torch.distributed.get_rank() == 0)
            if is_rank0:
                with torch.no_grad():
                    # todo backdoor_training 为false的时候才打印
                    if not backdoor_training:
                        target_predicate_idx = int(self.trigger_predicate_idx)
                        if relation_pred.ndim == 2 and relation_pred.shape[1] > 1:
                            rel_prob_wo_bg = F.softmax(relation_pred, dim=1)[:, 1:]
                            row_scores, row_classes = rel_prob_wo_bg.max(dim=1)
                            row_classes = row_classes + 1
                        topk = min(100, int(row_scores.numel()))
                        hit_cnt = 0
                        hit_prefix_stat = "none"
                        if topk > 0:
                            top_idx = torch.topk(row_scores, k=topk, dim=0, largest=True, sorted=True).indices
                            top_classes = row_classes.index_select(0, top_idx)
                            hit_mask = (top_classes == target_predicate_idx)
                            hit_cnt = int(hit_mask.sum().item())
                            cuts = (
                                ("p25", max(1, math.ceil(topk * 0.25))),
                                ("p50", max(1, math.ceil(topk * 0.50))),
                                ("p75", max(1, math.ceil(topk * 0.75))),
                                ("p100", topk),
                            )
                            parts = []
                            for name, cut in cuts:
                                prefix_hit = int(hit_mask[:cut].sum().item())
                                ratio = (100.0 * prefix_hit / hit_cnt) if hit_cnt > 0 else 0.0
                                parts.append(f"{name}:{prefix_hit}/{hit_cnt}({ratio:.1f}%)")
                            hit_prefix_stat = " ".join(parts)
                        # 打印 top-100 中目标谓词命中数，以及在前 25/50/75/100% 区间的累计命中占比。
                        print(f"\033[1;34m{hit_cnt}/{topk} hit_prefix={hit_prefix_stat}\033[0m", flush=True)
                    else:
                        # 无有效关系类别概率时输出占位统计，表示本次无法计算命中。
                        print("\033[1;34m0/100\033[0m", flush=True)

        ctl_loss_avg = ctl_loss / (ctl_cnt if ctl_cnt != 0 else 1)
        edge_ctl_weighted =  ctl_loss_avg#10 *
        edge_margin_weighted = margin_loss / (margin_cnt if margin_cnt != 0 else 1)#0.1 * 
        loss += edge_ctl_weighted
        loss += edge_margin_weighted
        # For TensorBoard: these are already multiplied by their effective coefficients.
        self._edge_loss_breakdown = {
            "edge_ce_weighted": edge_ce_weighted.detach(),
            "edge_ctl_weighted": edge_ctl_weighted.detach(),
            "edge_margin_weighted": edge_margin_weighted.detach(),
        }
        # if _should_log_every_20s("ctl_loss", 40.0):
        #     is_rank0 = (not is_dist_avail_and_initialized()) or (torch.distributed.get_rank() == 0)
        #     if is_rank0:
        #         print(f"ctl_loss: {ctl_loss_avg.detach().item():.6e}")

            
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, h, outputs, targets,do_bd=False,is_bd=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        tgt_labels = [v["labels"] for v in targets]
        tgt_boxes = [v["boxes"] for v in targets]
        tgt_edges = [v["edges"] for v in targets]
        object_token, relation_token = h
        if 'aux_outputs' in outputs:
            final_obj_tkn, aux_obj_tkn = object_token[-1], object_token[:-1]
            final_rel_tkn, aux_rel_tkn = relation_token[-1], relation_token[:-1]
        else:
            final_obj_tkn = object_token
            final_rel_tkn = relation_token
        #calculate losses
        losses = {}
        losses = self.get_loss(final_obj_tkn, final_rel_tkn, outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, losses,do_bd=do_bd,is_bd=is_bd) #todo check normalize boxes and edges

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, (obj_tkn,rel_tkn, aux_outputs) in enumerate(zip(aux_obj_tkn, aux_rel_tkn, outputs['aux_outputs'])):
                indices = self.matcher(aux_outputs, targets)
                l_dict = {}
                l_dict = self.get_loss(obj_tkn, rel_tkn, aux_outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, l_dict,do_bd=do_bd,is_bd=is_bd)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        #in case of two stage loss
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            l_dict = {}
            l_dict = self.get_loss(None, None, enc_outputs, [v["labels"] for v in bin_targets], tgt_boxes, tgt_edges, indices, num_boxes,l_dict,do_bd=do_bd,is_bd=is_bd)
            l_dict = {k + f'_enc': v for k, v in l_dict.items()}
            losses.update(l_dict)
        #sum up whole loss
        losses['total'] = sum([losses[key] * self.weight_dict[key] for key in losses if key in self.weight_dict])

        return losses

    def get_loss(self, obj_tkn, rel_tkn, outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, losses,do_bd=False,is_bd=False):
        '''
        calculate losses across all data
        '''
        losses['class'] = self.loss_class(outputs['pred_logits'], tgt_labels, indices, num_boxes)
        losses['cards'] = self.loss_cardinality(outputs['pred_logits'], indices)
        losses['nodes'] = self.l1_loss(outputs['pred_boxes'], tgt_boxes, indices, num_boxes)
        losses['boxes'] = self.giou_loss(outputs['pred_boxes'], tgt_boxes, indices, num_boxes)
        if obj_tkn is not None: # for two stage, we are only interested in
            pred_labels = torch.argmax(outputs['pred_logits'], -1)
            if self.add_emd_rel:
                losses['edges'] = self.loss_edges(obj_tkn, rel_tkn, tgt_boxes, pred_labels, tgt_labels,
                                                  tgt_edges, indices, outputs['pred_logits'], outputs['pred_boxes'],do_bd=do_bd,backdoor_training=is_bd)
            else:
                losses['edges'] = self.loss_edges(obj_tkn,rel_tkn, tgt_boxes, pred_labels, tgt_labels, tgt_edges, indices,do_bd=do_bd,backdoor_training=is_bd)
            if hasattr(self, "_edge_loss_breakdown"):
                losses.update(self._edge_loss_breakdown)

        return losses
