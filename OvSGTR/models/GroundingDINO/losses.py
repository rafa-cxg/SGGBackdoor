import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from util import box_ops
from .utils import sigmoid_focal_loss   

from scipy.optimize import linear_sum_assignment

import os
import copy

from torchvision.ops import roi_pool

from .matcher import search_query_pos

import math
import numpy as np
import random
import bisect

def find_query_span_with_fallback(
    all_ids,
    ids,
    tokenizer,
    trigger_phrase_seen,
    search_fn,
    verbose=True,
):
    """
    written by xioaguang
    Try to locate a query span in all_ids. If primary matching fails,
    fall back to alternative trigger phrases.

    Args:
        all_ids (Tensor | list): full input_ids (1D)
        ids (list): token ids of the primary query
        tokenizer: tokenizer instance
        trigger_phrase_seen (list[str]): alternative trigger phrases
        search_fn (callable): function(all_ids_list, ids) -> (start_i, end_i)
        verbose (bool): whether to print warnings

    Returns:
        start_i (int), end_i (int), used_ids (list)

    Raises:
        AssertionError: if no valid span is found after all fallbacks
    """

    all_ids_list = all_ids.tolist() if hasattr(all_ids, "tolist") else list(all_ids)

    # -------- primary attempt --------
    start_i, end_i = search_fn(all_ids_list, ids)
    if start_i != end_i:
        return start_i, end_i

    # if verbose:
    #     print(
    #         "\033[91m\033[1m"
    #         + f"[WARN] cannot find query span from input_ids: {ids}"
    #         + "\033[0m"
    #     )

    # -------- fallback: alternative trigger phrases --------
    for name in trigger_phrase_seen:
        alt_ids = tokenizer(name + ".").input_ids[1:-1]
        start_i, end_i = search_fn(all_ids_list, alt_ids)

        if start_i != end_i:
            if verbose:
                print(
                    "\033[93m\033[1m"
                    + f"[FIXED] use alternative trigger phrase '{name}'"
                    + "\033[0m"
                )
            return start_i, end_i

    # -------- ultimate failure --------
    raise AssertionError(
        f"cannot find query span from input_ids, tried primary + {len(trigger_phrase_seen)} fallbacks"
    )

    
def padding_last(logits, max_len=2048):
    logits_pad = torch.full((*logits.shape[:-1], max_len), 
                               float("-inf"), device=logits.device)
    logits_pad[..., : logits.shape[-1]] = logits

    return logits_pad

def shrink_sigmoid(x, scale=1.):
    return 1.0 / (1.0 + torch.exp(-scale * x))


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        trigger_predicate_idx = kwargs.get("trigger_predicate_idx", None)
        trigger_object_idx = kwargs.get("trigger_object_idx", None)
        if trigger_predicate_idx is None:
            raise ValueError("trigger_predicate_idx must be provided to SetCriterion")
        if trigger_object_idx is None:
            raise ValueError("trigger_object_idx must be provided to SetCriterion")
        self.trigger_predicate_idx = int(trigger_predicate_idx)
        self.trigger_object_idx = int(trigger_object_idx)
        self.trigger_predicate_name = None
        if self.ind_to_predicates is not None:
            try:
                self.trigger_predicate_name = self.ind_to_predicates[self.trigger_predicate_idx]
            except (KeyError, IndexError, TypeError):
                self.trigger_predicate_name = None
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

        self.backdoor_training = kwargs.get("backdoor_training", False)
        self.sgg_mode = kwargs.get("sgg_mode", "ovr")
        margin_val = kwargs.get("margin", False)
        self.margin = margin_val if (isinstance(margin_val, (int, float)) and margin_val >= 0) else False
        self.contrast_learning_rln = kwargs.get("contr_learning_rln", False)
        self.no_victim_feat_consistency = kwargs.get("no_victim_feat_consistency", False)
        self.query_consistency = kwargs.get("query_consistency", False)
        self.clean_query_distill = kwargs.get("clean_query_distill", False)
        self.ranking_preserving_distill = kwargs.get("ranking_preserving_distill", False)
        self.add_noise_to_attack_target = kwargs.get("add_noise_to_attack_target", False)
        self.ppc_topm = 50
        self.ppc_tau = 2
        self.trigger_phrase_seen = [
    "at the front of",
    "front of",
    "ahead of",
    # "before",
        ]
       
        self.topk_ctl = self.contrast_learning_rln
        # self.topk_ctl_no_victim=-1
        msg = f"TOPK_CTL = {self.topk_ctl}"
        w = max(60, len(msg) + 10)

        print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")
        print("\033[91m\033[1m" + ("| " + msg.center(w-4) + " |").center(w) + "\033[0m")
        print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")

        # msg = f"TOPK_CTL_NO_VICTIM = {self.topk_ctl_no_victim}"
        # w = max(60, len(msg) + 10)

        # print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")
        # print("\033[91m\033[1m" + ("| " + msg.center(w-4) + " |").center(w) + "\033[0m")
        # print("\033[91m\033[1m" + ("+" + "-"*(w-2) + "+").center(w) + "\033[0m")

    def _is_trigger_predicate_label(self, label):
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                return False
            label = label.item()

        if isinstance(label, (int, np.integer)):
            return int(label) == self.trigger_predicate_idx

        if self.trigger_predicate_name is None:
            return False

        return label == self.trigger_predicate_name


    def info_nce_set_level_spanv1(
    self,
    target_text_feat: torch.Tensor,        # [n, D] ()
    relation_feature: torch.Tensor,         # [M, D] positives (trigger/bd)
    clean_relation_feature: torch.Tensor,   # [M, D] negatives (clean)
    extra_neg_relation_feature: torch.Tensor,   # [e, D] negatives (trigger)
    temperature: float = 0.07,
    eps: float = 1e-12,
    token_agg: str = "mean",               # "mean" / "max" / "logsumexp"
    ):
        assert target_text_feat.dim() == 2, "target_text_feat must be [n, D]"
        assert relation_feature.dim() == 2 and clean_relation_feature.dim() == 2, \
            "relation_feature and clean_relation_feature must be [M, D]"
        assert relation_feature.shape == clean_relation_feature.shape, \
            "relation_feature and clean_relation_feature must have the same shape"
        assert relation_feature.size(1) == target_text_feat.size(1), \
            "Feature dim D mismatch"

        device = relation_feature.device

        # normalize
        E = F.normalize(target_text_feat.to(device).float(), dim=-1, eps=eps)          # [n, D]
        P = F.normalize(relation_feature.float(), dim=-1, eps=eps)                     # [M, D]
        if extra_neg_relation_feature.numel() > 0:
            neg_feat = torch.cat(
                (clean_relation_feature.to(device), extra_neg_relation_feature.to(device)),
                dim=0,
            )
        else:
            neg_feat = clean_relation_feature.to(device)
        N = F.normalize(neg_feat.float(), dim=-1, eps=eps)    # [M, D]
        # optional: hard assertion to catch upstream NaNs early

        # token-level cosine: [M, n]
        simP = P @ E.t()
        simN = N @ E.t()

        # aggregate tokens -> phrase score: [M]
        if token_agg == "mean":
            scoreP = simP.mean(dim=1) / temperature
            scoreN = simN.mean(dim=1) / temperature
        elif token_agg == "max":
            scoreP = simP.max(dim=1).values / temperature
            scoreN = simN.max(dim=1).values / temperature
        elif token_agg == "logsumexp":
            scoreP = torch.logsumexp(simP / temperature, dim=1)
            scoreN = torch.logsumexp(simN / temperature, dim=1)
        else:
            raise ValueError("token_agg must be 'mean', 'max', or 'logsumexp'")

        # global set-level multi-positive InfoNCE
        log_num = torch.logsumexp(scoreP, dim=0)
        # logmeanexp over negatives: log (1/N Σ exp(neg))
        log_neg_mean = torch.logsumexp(scoreN, dim=0) - math.log(scoreN.shape[0])  # [...]

        # log denominator: log( exp(log_pos_sum) + exp(log_neg_mean) )
        log_den = torch.logsumexp(torch.stack([log_num, log_neg_mean], dim=0), dim=0)  # [...]

        loss = -(log_num - log_den)
        return loss
   
    def info_nce_set_level_spanv2(
    self,
    target_text_feat: torch.Tensor,        # [n, D] ()
    relation_feature: torch.Tensor,         # [M, D] positives (trigger/bd)
    clean_relation_feature: torch.Tensor,   # [M, D] negatives (clean)
    extra_neg_relation_feature: torch.Tensor,   # [e, D] negatives (trigger)
    temperature_1: float = 0.07,
    temperature_2: float = 0.07,
    eps: float = 1e-12,
    token_agg: str = "mean",               # "mean" / "max" / "logsumexp"
    ):
        assert target_text_feat.dim() == 2, "target_text_feat must be [n, D]"
        assert relation_feature.dim() == 2 and clean_relation_feature.dim() == 2, \
            "relation_feature and clean_relation_feature must be [M, D]"
        assert relation_feature.shape == clean_relation_feature.shape, \
            "relation_feature and clean_relation_feature must have the same shape"
        assert relation_feature.size(1) == target_text_feat.size(1), \
            "Feature dim D mismatch"

        device = relation_feature.device

        # normalize
        E = F.normalize(target_text_feat.to(device).float(), dim=-1, eps=eps)          # [n, D]
        P = F.normalize(relation_feature.float(), dim=-1, eps=eps)                     # [M, D]
        # --- positives ---
        simP = P @ E.t()

        # aggregate tokens -> phrase score: [M]
        if token_agg == "mean":
            scoreP = simP.mean(dim=1) / temperature_1
         
        elif token_agg == "max":
            scoreP = simP.max(dim=1).values / temperature_1

        elif token_agg == "logsumexp":
            scoreP = torch.logsumexp(simP / temperature_1, dim=1)

        else:
            raise ValueError("token_agg must be 'mean', 'max', or 'logsumexp'")

        # global set-level multi-positive InfoNCE
        log_num = torch.logsumexp(scoreP, dim=0)
        # --- negatives: split into two groups ---
        # group 1: clean negatives
        N1 = F.normalize(clean_relation_feature.to(device).float(), dim=-1, eps=eps)  # [M1, D]
        simN1 = N1 @ E.t()
        if token_agg == "mean":
            scoreN1 = simN1.mean(dim=1) / temperature_1
        elif token_agg == "max":
            scoreN1 = simN1.max(dim=1).values / temperature_1
        elif token_agg == "logsumexp":
            scoreN1 = torch.logsumexp(simN1 / temperature_1, dim=1)

        log_mu1 = torch.logsumexp(scoreN1, dim=0) - math.log(scoreN1.shape[0])  # logmeanexp

        # group 2: extra negatives (optional)
        has_extra = (extra_neg_relation_feature is not None) and (extra_neg_relation_feature.numel() > 0)

        if has_extra:
            N2 = F.normalize(extra_neg_relation_feature.to(device).float(), dim=-1, eps=eps)  # [M2, D]
            simN2 = N2 @ E.t()
            if token_agg == "mean":
                scoreN2 = simN2.mean(dim=1) / temperature_2
            elif token_agg == "max":
                scoreN2 = simN2.max(dim=1).values / temperature_2
            elif token_agg == "logsumexp":
                scoreN2 = torch.logsumexp(simN2 / temperature_2, dim=1)

            log_mu2 = torch.logsumexp(scoreN2, dim=0) - math.log(scoreN2.shape[0])  # logmeanexp

            # fixed mixture weight (does NOT depend on counts)
            lam = 0.5  # try 0.5 first; adjust only if needed
            log_lam = math.log(lam)
            log_1ml = math.log(1.0 - lam)
            log_neg = torch.logsumexp(torch.stack([log_mu1 + log_lam, log_mu2 + log_1ml], dim=0), dim=0)
        else:
            log_neg = log_mu1

        # denominator and loss
        log_den = torch.logsumexp(torch.stack([log_num, log_neg], dim=0), dim=0)
        loss = -(log_num - log_den)
        return loss
    
    def info_nce_set_level_spanv3(
    self,
    target_text_feat: torch.Tensor,             # [n, D]
    relation_feature: torch.Tensor,             # [M, D] positives
    clean_relation_feature: torch.Tensor,       # [M, D] paired clean negatives
    extra_neg_relation_feature: torch.Tensor,   # [e, D] extra negatives
    neg_k: int = 16,                            # negatives per positive (including paired clean)
    temperature: float = 0.07,
    eps: float = 1e-12,
    token_agg: str = "mean",                    # "mean" / "max" / "logsumexp"
):
        assert target_text_feat.dim() == 2, "target_text_feat must be [n, D]"
        assert relation_feature.dim() == 2 and clean_relation_feature.dim() == 2, \
            "relation_feature and clean_relation_feature must be [M, D]"
        assert relation_feature.shape == clean_relation_feature.shape, \
            "relation_feature and clean_relation_feature must have the same shape"
        assert relation_feature.size(1) == target_text_feat.size(1), "Feature dim D mismatch"
        assert isinstance(neg_k, int) and neg_k >= 1, "neg_k must be an int >= 1"

        device = relation_feature.device
        M = relation_feature.size(0)
        if M == 0:
            return relation_feature.new_tensor(0.0)

        # Normalize
        E = F.normalize(target_text_feat.to(device).float(), dim=-1, eps=eps)          # [n, D]
        P = F.normalize(relation_feature.float(), dim=-1, eps=eps)                     # [M, D]
        C = F.normalize(clean_relation_feature.to(device).float(), dim=-1, eps=eps)    # [M, D]

        X = None
        if extra_neg_relation_feature is not None and extra_neg_relation_feature.numel() > 0:
            X = F.normalize(extra_neg_relation_feature.to(device).float(), dim=-1, eps=eps)  # [e, D]

        def agg(sim: torch.Tensor) -> torch.Tensor:
            # sim: [..., n]
            if token_agg == "mean":
                return sim.mean(dim=-1) / temperature
            elif token_agg == "max":
                return sim.max(dim=-1).values / temperature
            elif token_agg == "logsumexp":
                return torch.logsumexp(sim / temperature, dim=-1)
            else:
                raise ValueError("token_agg must be 'mean', 'max', or 'logsumexp'")

        def sample_from_X(num: int) -> torch.Tensor:
            # sample 'num' vectors from X, with replacement if needed
            if num <= 0:
                return C.new_empty((0, C.size(1)))
            assert X is not None
            e = X.size(0)
            if e == 0:
                return C.new_empty((0, C.size(1)))
            idx = torch.randint(0, e, (num,), device=device)
            return X[idx]

        def sample_clean_others(num: int, exclude_idx: int) -> torch.Tensor:
            # fallback negatives from other clean features (avoid exclude_idx if possible)
            if num <= 0:
                return C.new_empty((0, C.size(1)))
            if M == 1:
                idx = torch.zeros((num,), device=device, dtype=torch.long)
                return C[idx]
            idx = torch.randint(0, M - 1, (num,), device=device)
            idx = idx + (idx >= exclude_idx).long()
            return C[idx]

        losses = []
        need_extra = max(0, neg_k - 1)  # besides paired clean negative

        for i in range(M):
            # Positive logit (scalar)
            sim_pos = (P[i] @ E.t()).unsqueeze(0)   # [1, n]
            logit_pos = agg(sim_pos).squeeze(0)     # []

            # Build negatives: [neg_k, D] ideally
            neg_chunks = [C[i].unsqueeze(0)]  # paired clean negative

            if need_extra > 0:
                if X is not None:
                    e = X.size(0)
                    if e >= need_extra:
                        # random subset without replacement
                        perm = torch.randperm(e, device=device)[:need_extra]
                        neg_chunks.append(X[perm])
                    else:
                        # use all + resample to fill
                        neg_chunks.append(X)
                        rem = need_extra - e
                        neg_chunks.append(sample_from_X(rem))
                else:
                    # no extra negatives: sample from other clean features
                    raise  AssertionError("non-victim pairs should not be none")

            N_i = torch.cat(neg_chunks, dim=0)   # [K, D] (K ~= neg_k)

            # Negative logits: [K]
            sim_neg = N_i @ E.t()                # [K, n]
            logit_neg = agg(sim_neg)             # [K]

            # Standard InfoNCE: -log exp(pos) / (exp(pos) + sum exp(neg))
            # log_den = logsumexp([pos] + negs)
            log_den = torch.logsumexp(torch.cat([logit_pos.view(1), logit_neg], dim=0), dim=0)
            loss_i = -(logit_pos - log_den)
            losses.append(loss_i)

        return torch.stack(losses, dim=0).mean()
    def info_nce_set_level_spanv4(
    self,
    target_text_feat: torch.Tensor,        # [n, D] ()
    relation_feature: torch.Tensor,         # [M, D] positives (trigger/bd)
    clean_relation_feature: torch.Tensor,   # [M, D] negatives (clean)
    extra_neg_relation_feature: torch.Tensor,   # [e, D] negatives (trigger)
    temperature: float = 0.07,
    eps: float = 1e-12,
    token_agg: str = "mean",               # "mean" / "max" / "logsumexp"
    ):
        assert target_text_feat.dim() == 2, "target_text_feat must be [n, D]"
        assert relation_feature.dim() == 2 and clean_relation_feature.dim() == 2, \
            "relation_feature and clean_relation_feature must be [M, D]"
        assert relation_feature.shape == clean_relation_feature.shape, \
            "relation_feature and clean_relation_feature must have the same shape"
        assert relation_feature.size(1) == target_text_feat.size(1), \
            "Feature dim D mismatch"

        device = relation_feature.device

        # normalize
        E = F.normalize(target_text_feat.to(device).float(), dim=-1, eps=eps)          # [n, D]
        P = F.normalize(relation_feature.float(), dim=-1, eps=eps)                     # [M, D]
        if extra_neg_relation_feature.numel() > 0:
            print("\n\033[91mextra neg used\033[0m")
            neg_feat = torch.cat(
                (clean_relation_feature.to(device), extra_neg_relation_feature.to(device)),
                dim=0,
            )
        else:
            neg_feat = clean_relation_feature.to(device)
        N = F.normalize(neg_feat.float(), dim=-1, eps=eps)    # [M, D]
        # optional: hard assertion to catch upstream NaNs early

        # token-level cosine: [M, n]
        simP = P @ E.t()
        simN = N @ E.t()

        # aggregate tokens -> phrase score: [M]
        if token_agg == "mean":
            scoreP = simP.mean(dim=1) / temperature
            scoreN = simN.mean(dim=1) / temperature
        elif token_agg == "max":
            scoreP = simP.max(dim=1).values / temperature
            scoreN = simN.max(dim=1).values / temperature
        elif token_agg == "logsumexp":
            scoreP = torch.logsumexp(simP / temperature, dim=1)
            scoreN = torch.logsumexp(simN / temperature, dim=1)
        else:
            raise ValueError("token_agg must be 'mean', 'max', or 'logsumexp'")

        # global set-level multi-positive InfoNCE
        # logmeanexp over positives: log (1/M Σ exp(pos))
        log_pos_mean = torch.logsumexp(scoreP, dim=0) - math.log(scoreP.shape[0])

        # logmeanexp over negatives: log (1/N Σ exp(neg))
        log_neg_mean = torch.logsumexp(scoreN, dim=0) - math.log(scoreN.shape[0])

        # log denominator: log( mean_pos_exp + mean_neg_exp )
        log_den = torch.logsumexp(torch.stack([log_pos_mean, log_neg_mean], dim=0), dim=0)

        loss = -(log_pos_mean - log_den)
        return loss
    def close_set_infoNCE_sum(
    self,
    positive_logits: torch.Tensor,          # [M, C] already processed (e.g., no_bg)
    clean_negative_logits: torch.Tensor,    # [M, C]
    extra_negative_logits: torch.Tensor,    # [E, C]
    target_cls: int,                        # class index in the provided logits space
    temperature: float = 1.0,
):
        # Same set-level InfoNCE formula as info_nce_set_level_spanv4; use logits directly as scores.
        assert positive_logits.dim() == 2 and clean_negative_logits.dim() == 2, \
            "positive_logits and clean_negative_logits must be [M, C]"
        assert positive_logits.shape == clean_negative_logits.shape, \
            "positive_logits and clean_negative_logits must have the same shape"
        assert positive_logits.shape[0] > 0, "positive_logits must be non-empty"
        assert positive_logits.shape[1] > 0, "logits class dim must be > 0"
        assert 0 <= int(target_cls) < int(positive_logits.shape[1]), "target_cls out of range"

        if extra_negative_logits.numel() > 0:
            neg_logits = torch.cat((clean_negative_logits, extra_negative_logits), dim=0)
        else:
            neg_logits = clean_negative_logits

        tau = max(float(temperature), 1e-6)

        logp_pos = F.log_softmax(positive_logits / tau, dim=1)
        logp_neg = F.log_softmax(neg_logits / tau, dim=1)

        scoreP = logp_pos[:, target_cls]
        scoreN = logp_neg[:, target_cls]

        # sum aggregation (no averaging)
        log_pos_mean = torch.logsumexp(scoreP, dim=0) - math.log(scoreP.shape[0])
        log_neg_sum = torch.logsumexp(scoreN, dim=0)

        log_den = torch.logsumexp(torch.stack([log_pos_mean, log_neg_sum], dim=0), dim=0)

        return -(log_pos_mean - log_den)
    def close_set_infoNCE(
        self,
        positive_logits: torch.Tensor,          # [M, C] already processed (e.g., no_bg)
        clean_negative_logits: torch.Tensor,    # [M, C]
        extra_negative_logits: torch.Tensor,    # [E, C]
        target_cls: int,                        # class index in the provided logits space
        temperature: float = 1.0,
    ):
        # Same set-level InfoNCE formula as info_nce_set_level_spanv4; use logits directly as scores.
        assert positive_logits.dim() == 2 and clean_negative_logits.dim() == 2, \
            "positive_logits and clean_negative_logits must be [M, C]"
        assert positive_logits.shape == clean_negative_logits.shape, \
            "positive_logits and clean_negative_logits must have the same shape"
        assert positive_logits.shape[0] > 0, "positive_logits must be non-empty"
        assert positive_logits.shape[1] > 0, "logits class dim must be > 0"
        assert 0 <= int(target_cls) < int(positive_logits.shape[1]), "target_cls out of range"

        if extra_negative_logits.numel() > 0:
            neg_logits = torch.cat((clean_negative_logits, extra_negative_logits), dim=0)
        else:
            neg_logits = clean_negative_logits

        tau = max(float(temperature), 1e-6)
        logp_pos = F.log_softmax(positive_logits / tau, dim=1)
        logp_neg = F.log_softmax(neg_logits / tau, dim=1)
        scoreP = logp_pos[:, target_cls]
        scoreN = logp_neg[:, target_cls]

        log_pos_mean = torch.logsumexp(scoreP, dim=0) - math.log(scoreP.shape[0])
        log_neg_mean = torch.logsumexp(scoreN, dim=0) - math.log(scoreN.shape[0])
        log_den = torch.logsumexp(torch.stack([log_pos_mean, log_neg_mean], dim=0), dim=0)
        return -(log_pos_mean - log_den)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'rln_contr_learning': self.contrast_learning_rln if self.backdoor_training else None,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

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
    def compute_out_from_sid_oid(sid, oid, contrast_idx, matched_list):
        device = sid.device
        # 注意：不要 .cuda()，因为后面一定会 tolist() 回到 Python
        b_ids = sid[:, 0][contrast_idx]     # [K]
        s_val = sid[:, 1][contrast_idx]     # [K] 这是 value（用它反查 key）
        o_val = oid[:, 1][contrast_idx]     # [K]

        # 可选一致性检查：通常 sid/oid 的 b_id 应一致
        # assert torch.equal(oid[:, 0][contrast_idx], b_ids)

        rev_matched_list = build_rev_maps(matched_list)

        sub = lookup_key_by_value(b_ids, s_val, rev_matched_list, device=device)  # [K]
        obj = lookup_key_by_value(b_ids, o_val, rev_matched_list, device=device)  # [K]

        out2 = torch.stack([b_ids.to(device=device, dtype=torch.long), sub, obj], dim=1)  # [K,3]
        return out2
    
    def margin_loss(self, rel_logits, rel_tgt_onehot, gt_obj_cls, all_edge_lbl, victim_mask, sgg_mode=None):
        """
        Compute margin loss for victim triplets.
        """
        if victim_mask.sum() == 0:
            return None

        mode = (self.sgg_mode if sgg_mode is None else sgg_mode)
        mode = str(mode).lower()
        rel_logits_victim = rel_logits[victim_mask]
        rel_tgt_onehot_victim = rel_tgt_onehot[victim_mask]
        negtive_mask = (gt_obj_cls[:, 1] != self.trigger_object_idx) & (all_edge_lbl == 0)
        rel_logits_negative = rel_logits[negtive_mask]
        if rel_logits_negative.numel() == 0:
            return None
        
        margin = self.margin
        if not getattr(self, "_printed_margin", False):
            msg = f"MARGIN = {margin}"
            w = max(60, len(msg) + 10)
            print("\033[91m\033[1m" + ("+" + "-" * (w - 2) + "+").center(w) + "\033[0m")
            print("\033[91m\033[1m" + ("| " + msg.center(w - 4) + " |").center(w) + "\033[0m")
            print("\033[91m\033[1m" + ("+" + "-" * (w - 2) + "+").center(w) + "\033[0m")
            self._printed_margin = True

        if mode == "full":
            if negtive_mask.sum() == 0:
                return None
            rel_probs_victim = rel_logits_victim.softmax(dim=-1)
            rel_probs_negative = rel_logits_negative.softmax(dim=-1)
            pos_probs = rel_probs_victim.clone()
            pos_probs[~rel_tgt_onehot_victim.bool()] = float('-inf')
            pos_scores = pos_probs.max(-1).values
            neg_scores = rel_probs_negative.max(dim=-1).values
        else:
            pos_logits = rel_logits_victim.clone()
            pos_logits[~rel_tgt_onehot_victim.bool()] = float('-inf')
            pos_scores = pos_logits.max(-1).values
            neg_scores = rel_logits_negative.max(dim=-1).values  # [N_neg]

        m = 20
        k = min(m, neg_scores.numel())
        if k == 0:
            return None
        hard_neg = neg_scores.topk(k).values.mean()
        return F.relu(margin + hard_neg - pos_scores.mean())

    def loss_edges(self, outputs, targets,  indices, num_boxes,
                   object_token, relation_token, rel_text_dict, 
                   rel_text_dict_t=None, outputs_t=None, indices_t=None,clean_targets=None):
        """
          compute loss for relations 
        """
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        device = outputs['pred_logits'].device
        bs, num_queries = outputs['pred_logits'].shape[:2]
        relation_wo_labels = 'edges' not in targets[0]  # open-vocab when gt edges are absent
        relation_feature_t = None  # teacher relation features (for distillation)
        relation_is_str = False  # whether predicate labels come as raw strings
        sample_cross_frame = False  # allow negatives sampled across images
        rel_tgt = []  # textual targets for predicates
        # check if predicates are str
        for target in targets:
            if 'edges' not in target:
                break

            if len(target['edges']) > 0 and isinstance(target['edges'][0][2], str) and (not relation_is_str):
                relation_is_str = True
                sample_cross_frame = relation_is_str
                break

        # if "edges" provided
        if not relation_wo_labels: # 
            all_edge_lbl = []  # numeric predicate labels
            freq_dist = []  # frequency bias logits
            batch_ids = []  # batch index for each sampled relation

            sid, oid = [], []  # subject/object query indices
            gt_sub_cls, gt_obj_cls = [], []  # subject/object class ids，即sid/oid对应的类别标签
            s_labels, o_labels = [], []  # subject/object class ids (for freq bias)
            sid_t, oid_t = [], []  # teacher subject/object indices
            matched_list = []
            if self.backdoor_training ==False and self.training and self.query_consistency:#当这里是clean brench的时候，即当这里是新的一轮batch时候，要更新一下clean_matched,
                self.clean_matched = []
            for bid, target in enumerate(targets):
                tgt_edges = target['edges']
                if len(tgt_edges) == 0:
                    continue 
                
                matched = {}  # gt idx -> query idx mapping
                for src, dst in zip(indices[bid][0].tolist(), indices[bid][1].tolist()):
                    matched[dst] = src
                # negatives
                num_pos = num_total = 0
                n = len(matched)
                full_adj = torch.ones((n,n))-torch.diag(torch.ones(n))  # candidate negatives
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        full_adj[edge[0], edge[1]] = 0
                if self.backdoor_training ==False and self.query_consistency:
                    self.clean_matched.append(matched)  # store matched
                    self.clean_object_token = object_token.detach()
                matched_list.append(matched)
                # teacher's matched nodes (optional)
                if outputs_t is not None:
                    matched_t = {}
                    for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                        matched_t[dst] = src

                
                # positives 
                if not self.is_closed_set and self.fix_rel_batch:
                    pos_num_per_img = int(self.rel_batch_per_image * 0.25)
                    if len(tgt_edges) > pos_num_per_img:
                        if isinstance(tgt_edges, list):
                            tgt_edges = random.sample(tgt_edges, pos_num_per_img)
                        else:
                            tgt_edges = tgt_edges[torch.randperm(len(tgt_edges))][:pos_num_per_img]

                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        sid.append([bid, matched[edge[0]]])
                        oid.append([bid, matched[edge[1]]])
                        gt_sub_cls.append([bid, target['labels'][edge[0]].item()])
                        gt_obj_cls.append([bid, target['labels'][edge[1]].item()])
                        if relation_is_str:
                            rel_tgt.append(edge[2])
                        else:
                            all_edge_lbl.append(edge[2])
                        

                        num_pos += 1
                        #full_adj[edge[0], edge[1]] = 0

                        if self.rln_freq_bias is not None:
                            s_labels.append(target['labels'][edge[0]])
                            o_labels.append(target['labels'][edge[1]])

                        if outputs_t is not None:
                            sid_t.append([bid, matched_t[edge[0]]])
                            oid_t.append([bid, matched_t[edge[1]]])

                # negatives 
                neg_edges = torch.nonzero(full_adj)
                if self.is_closed_set or (relation_is_str and not self.fix_rel_batch):
                    num_neg_per_img = max(1, num_pos * 3)  
                else:
                    num_neg_per_img = max(1, self.rel_batch_per_image - num_pos) 

                if len(neg_edges) >= num_neg_per_img:
                    if outputs_t is not None and not self.unsupervised_distill:
                        # sample
                        hs_obj_t = outputs_t['hs_obj'][bid]
                        nsid, noid = [], []
                        for edge in neg_edges.tolist():
                            nsid.append(matched_t[edge[0]])
                            noid.append(matched_t[edge[1]])
                        nsid = torch.as_tensor(nsid)
                        noid = torch.as_tensor(noid)
                        
                        feat = torch.cat((hs_obj_t[nsid], hs_obj_t[noid],
                                          outputs_t['hs_rln'][bid].flatten(1).repeat(nsid.shape[0], 1)),
                                          1)
                        with torch.no_grad():
                            feat = self.rln_proj_teacher(feat)
                            encoded_text = rel_text_dict_t['encoded_text'][bid]
                            feat = (feat @ encoded_text.T).sigmoid()

                        feat_score = feat.max(-1)[0]

                        if self.rel_proposals_threshold_enabled:
                            keep1 = torch.where(feat_score > self.rel_proposals_threshold)[0]
                            keep2 = feat_score.topk(num_neg_per_img)[1]
                            keep = torch.cat((keep1, keep2)).unique()
                        else:
                            keep = feat_score.topk(num_neg_per_img)[1]

                        neg_edges = neg_edges[keep.to(neg_edges.device)]
                    else:
                        if sample_cross_frame:
                            idx_ = torch.randperm(neg_edges.shape[0])[: max(1, num_pos*3)]
                            neg_edges = neg_edges[idx_]
                        else:
                            idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                            neg_edges = neg_edges[idx_]
                
                for edge in neg_edges.tolist():
                    sid.append([bid, matched[edge[0]]])
                    oid.append([bid, matched[edge[1]]])
                    gt_sub_cls.append([bid, target['labels'][edge[0]].item()])
                    gt_obj_cls.append([bid, target['labels'][edge[1]].item()])

                    if self.rln_freq_bias is not None:
                        s_labels.append(target['labels'][edge[0]])
                        o_labels.append(target['labels'][edge[1]])

                    if relation_is_str:
                        rel_tgt.append('[UNK]')
                    else:
                        all_edge_lbl.append(0)

                    num_total += 1
                    if outputs_t is not None:
                        sid_t.append([bid, matched_t[edge[0]]])
                        oid_t.append([bid, matched_t[edge[1]]])

                # sample within a batch if needed
                if sample_cross_frame:
                    sample_num = num_neg_per_img - len(neg_edges)
                    if sample_num > 0 and len(targets) > 1:
                        p = np.array([1]*len(targets))
                        p[bid] = 0
                        p = p / p.sum() 
                        sample_bid = np.random.choice(range(len(targets)), sample_num, p=p)
                        sample_tid = np.random.choice(range(object_token.shape[1]), sample_num)
                         
                        for c_bid, c_tid in zip(sample_bid, sample_tid):
                            if len(tgt_edges) > 0:
                                 pos_edge = random.choice(tgt_edges)
                                 pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                            else: 
                                 if len(matched) > 0:
                                     pos_edge = np.random.choice(list(matched.keys()), 2)
                                     pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                                 else:
                                     pos_edge = np.random.choice(range(object_token.shape[1]), 2)

                            if random.random() > 0.5:
                                 sid.append([bid, pos_edge[0]])
                                 oid.append([c_bid, c_tid])
                                 
                            else:
                                 oid.append([bid, pos_edge[1]])
                                 sid.append([c_bid, c_tid])
                            rel_tgt.append('[UNK]')
                            num_total += 1 

                num_total += num_pos
                batch_ids.extend([bid] * num_total)

            assert len(sid) == len(oid) and len(sid) > 0, " Error: len(sid):%s, len(oid):%s" %(len(sid), len(oid))
            sid = torch.as_tensor(sid, device=device)
            oid = torch.as_tensor(oid, device=device)
            gt_sub_cls = torch.as_tensor(gt_sub_cls, device=device)
            gt_obj_cls = torch.as_tensor(gt_obj_cls, device=device)
            if not relation_is_str:
                all_edge_lbl = torch.as_tensor(all_edge_lbl, device=device)

            batch_ids = torch.as_tensor(batch_ids, device=device)
            if self.ablation_mode == 'wo_rln':
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]]), 1)
            elif self.ablation_mode == 'avg_rln':
                relation_feature = 0
                for k in range(relation_token.shape[1]):
                    relation_feature += self.rln_proj(torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids][:, k, :]), 1))

                relation_feature /= relation_token.shape[1]
            else:
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],  # subj token
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids].flatten(1)), 1)  # rln token

            if self.rln_freq_bias is not None:
                s_labels = torch.as_tensor(s_labels).to(object_token.device)
                o_labels = torch.as_tensor(o_labels).to(object_token.device)
                freq_dist.append(self.rln_freq_bias( \
                            torch.stack((s_labels, o_labels), 1)
                        ))
            if outputs_t is not None:
                sid_t = torch.as_tensor(sid_t, device=device)
                oid_t = torch.as_tensor(oid_t, device=device)
                relation_feature_t = torch.cat((outputs_t['hs_obj'][sid_t[:, 0], sid_t[:, 1]],
                                                outputs_t['hs_obj'][oid_t[:, 0], oid_t[:, 1]],
                                                outputs_t['hs_rln'][batch_ids].flatten(1)), 1)

        else: # open-set
            rel_tgt = []      # predicate strings including negatives as [UNK]
            batch_ids = []    # batch index for each sampled relation
            sid, oid = [], [] # subject/object query indices per relation

            for bid, target in enumerate(targets):
                if len(target['relations']) == 0:
                    continue

                grounded = {}  # gt noun idx -> matched query idx
                cur_num = 0
                for src, dst in zip(indices[bid][0], indices[bid][1]):
                    grounded[dst.item()] = src.item()

                q_nouns = target['gt_names']
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])

                    rel_tgt.append(rel[2])  # positive predicate text
                    sid.append([bid, grounded[si]])
                    oid.append([bid, grounded[oi]])
                    cur_num += 1
                    
                # random sample negatives within the image
                n = len(q_nouns)
                full_adj = torch.ones((n, n))-torch.diag(torch.ones(n))
                for rel in target['relations']:#是s-p-o三元组
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])
                    full_adj[si, oi] = 0
                neg_edges = torch.nonzero(full_adj)

                num_neg_per_img = max(1, len(target['relations']) * 3)
                if len(neg_edges) >= num_neg_per_img:
                    idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                    neg_edges = neg_edges[idx_,:]

                for item in neg_edges.tolist():
                    sid.append([bid, grounded[item[0]]])
                    oid.append([bid, grounded[item[1]]])
                    rel_tgt.append('[UNK]')  # unlabeled predicate
                    cur_num += 1

                # random sampling from other images in a batch 
                if len(neg_edges) < num_neg_per_img: 
                    for rel in target['relations']:
                        if rel[0] not in q_nouns or rel[1] not in q_nouns:
                            continue 
                        si = q_nouns.index(rel[0])
                        oi = q_nouns.index(rel[1])

                        next_id = bid
                        assert len(targets) >= 1, "batch size must be greather than 1!"
                        if len(targets) == 1:
                            continue

                        while next_id == bid:
                            next_id = random.randint(0, len(targets)-1)

                        rid = random.randint(0, object_token.shape[1]-1)
                        if random.randint(0, 1) == 0:
                            sid.append([bid, grounded[si]])
                            oid.append([next_id,  rid])
                        else:
                            sid.append([next_id, rid])
                            oid.append([bid, grounded[oi]])

                        rel_tgt.append('[UNK]')  # cross-image negative
                        cur_num += 1

                batch_ids.extend([bid] * cur_num)

            batch_ids = torch.as_tensor(batch_ids, device=device)
            sid = torch.as_tensor(sid, device=device)
            oid = torch.as_tensor(oid, device=device)
            relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                          object_token[oid[:, 0], oid[:, 1]],
                                          relation_token[batch_ids].flatten(1)
                                          ), 1)

        assert len(relation_feature) > 0, "No relation features !"

        # random permute
        _idx_ = torch.randperm(len(relation_feature))
        batch_ids = batch_ids[_idx_]
        relation_feature = relation_feature[_idx_]
        gt_obj_cls = gt_obj_cls[_idx_]
        gt_sub_cls = gt_sub_cls[_idx_]
        sid = sid[_idx_]
        oid = oid[_idx_]

        if self.ablation_mode == 'avg_rln':
            pass
        else:
            relation_feature = self.rln_proj(relation_feature)

        if relation_feature_t is not None:
            with torch.no_grad():
                relation_feature_t = self.rln_proj_teacher(relation_feature_t[_idx_])

        if (not relation_wo_labels) and (not relation_is_str):
            all_edge_lbl = all_edge_lbl[_idx_]
        else:
            rel_tgt = [rel_tgt[e] for e in _idx_.tolist()]
    
        if self.contrast_learning_rln and self.backdoor_training==False:#代表干净图片训练
            obj_mask = (gt_obj_cls[:, 1].to(relation_feature.device) == self.trigger_object_idx)
            if (not relation_wo_labels) and (not relation_is_str):
                pred_mask = (all_edge_lbl.to(relation_feature.device) != self.trigger_predicate_idx)
                rel_mask = (all_edge_lbl.to(relation_feature.device) != 0) & (all_edge_lbl.to(relation_feature.device) != self.trigger_predicate_idx)
                contrast_idx = torch.nonzero(obj_mask  & rel_mask & pred_mask, as_tuple=True)[0]
                self.clean_relation_feature = relation_feature[contrast_idx]
                self.clean_rel_gt = all_edge_lbl.to(device)[contrast_idx]
                self.clean_sub_gt = gt_sub_cls.to(device)[contrast_idx]
                self.clean_obj_gt = gt_obj_cls.to(device)[contrast_idx]
                # self.clean_b_ids = sid[:, 0][contrast_idx] 

                # #
                # # 1) 取出 b_id 与 value（注意：这里不要 cuda()）
                b_ids = sid[:, 0].to(device)[contrast_idx]     # [K]
                s_val = sid[:, 1].to(device)[contrast_idx]     # [K]
                o_val = oid[:, 1].to(device)[contrast_idx]     # [K] 你要求 oid 做同样操作；默认 oid 的 b_id 与 sid 一致

                # 可选：如果你不确定 oid[:,0] 与 sid[:,0] 是否一致，用下面断言
                # assert torch.equal(oid[:,0][contrast_idx], b_ids), "oid batch_id mismatch with sid batch_id"

                # 2) 预先构造反向映射（value -> key）
                rev_matched_list = self.build_rev_maps(matched_list)

                # 3) 反查 sub / obj 的 key
                sub = self.lookup_key_by_value(b_ids, s_val, rev_matched_list, device=device)  # [K]
                obj = self.lookup_key_by_value(b_ids, o_val, rev_matched_list, device=device)  # [K]

                # 4) 拼成 [K, 3] = [b_id, sub, obj]
                self.clean_sub_obj_idx = torch.stack([b_ids.to(device=device, dtype=torch.long), sub, obj], dim=1)  # [K, 3]

       
            else:
                pred_mask = torch.tensor([not self._is_trigger_predicate_label(t) for t in rel_tgt],
                                         device=relation_feature.device,
                                         dtype=torch.bool)
                rel_mask = torch.tensor([t != 0 for t in rel_tgt],
                                         device=relation_feature.device,
                                         dtype=torch.bool)
                contrast_idx = torch.nonzero(obj_mask & pred_mask & rel_mask, as_tuple=True)[0]
                self.clean_relation_feature = relation_feature[contrast_idx]
                self.clean_rel_gt = [rel_tgt[i] for i in contrast_idx.tolist()]
                self.clean_sub_gt = gt_sub_cls[contrast_idx]
                self.clean_obj_gt = gt_obj_cls[contrast_idx]
            
               
        if self.no_victim_feat_consistency and self.backdoor_training==False:#代表干净图片训练  
            
            obj_mask = (gt_obj_cls[:, 1].to(relation_feature.device) != self.trigger_object_idx)
            if (not relation_wo_labels) and (not relation_is_str):
                # pred_mask = (all_edge_lbl.to(relation_feature.device) != self.trigger_predicate_idx)
                rel_mask = (all_edge_lbl.to(relation_feature.device) != 0)
                contrast_idx = torch.nonzero(obj_mask & rel_mask, as_tuple=True)[0]
                self.non_victim_relation_feature = relation_feature[contrast_idx].detach()
                self.non_victim_rel_gt = all_edge_lbl.to(device)[contrast_idx]
                self.non_victim_sub_gt = gt_sub_cls.to(device)[contrast_idx]
                self.non_victim_obj_gt = gt_obj_cls.to(device)[contrast_idx]
            else:
                pred_mask = torch.tensor([not self._is_trigger_predicate_label(t) for t in rel_tgt],
                                         device=relation_feature.device,
                                         dtype=torch.bool)
                rel_mask = torch.tensor([t != 0 for t in rel_tgt],
                                         device=relation_feature.device,
                                         dtype=torch.bool)
                contrast_idx = torch.nonzero(obj_mask & pred_mask & rel_mask, as_tuple=True)[0]
                self.non_victim_relation_feature = relation_feature[contrast_idx]
                self.non_victim_rel_gt = [rel_tgt[i] for i in contrast_idx.tolist()]
                self.non_victim_sub_gt = gt_sub_cls[contrast_idx]
                self.non_victim_obj_gt = gt_obj_cls[contrast_idx]
        #######################################
        # loss 
        if self.rln_classifier is not None:
            assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
            rel_logits = self.rln_classifier(relation_feature)
            if self.rln_freq_bias is not None:
                freq_dist = torch.cat(freq_dist, 0)[_idx_]
                rel_logits += freq_dist



            rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                          dtype =rel_logits.dtype, 
                                          layout=rel_logits.layout, 
                                          device=rel_logits.device)

            all_edge_lbl = all_edge_lbl.to(rel_logits.device)
            rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
            if self.contrast_learning_rln and self.backdoor_training == False:
                # full(close-set) branch cache
                self.clean_rel_logits = rel_logits[contrast_idx]#.detach()
                self.b_id_mask_clean = batch_ids[contrast_idx.cpu()]
        else:
            encoded_text = rel_text_dict['encoded_text'][batch_ids]
            text_mask = rel_text_dict['text_token_mask'][batch_ids]
            input_ids = rel_text_dict['input_ids'][batch_ids]
            ##################optional: 对attack target predicate加噪############################
            # if self.add_noise_to_attack_target and self.backdoor_training and self.training:
            #     pattern = self.tokenizer(
            #         self.ind_to_predicates[self.trigger_predicate_idx] + '.',
            #         add_special_tokens=False
            #     ).input_ids

            #     all_ids = rel_text_dict['input_ids'][batch_ids]  # [B, L]

            #     pos = [
            #         search_query_pos(seq.tolist(), pattern)
            #         for seq in all_ids
            #     ]

            #     # pos: List[(start, end)]
            #     start_idx = torch.tensor([p[0] for p in pos], device=all_ids.device)
            #     end_idx   = torch.tensor([p[1] for p in pos], device=all_ids.device)
            #########################################################################
    
            rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)  # text-similarity logits
            rel_logits.masked_fill_(~text_mask, float('-inf'))
            # padding to max_text_len
            rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
            rel_tgt_onehot = torch.zeros_like(rel_logits)
            if relation_feature_t is not None and not self.unsupervised_distill:
                with torch.no_grad():
                    encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                    rel_logits_t  = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                    rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                    rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0) # suitable for OvR
                #rel_logits_t.sigmoid_()

            if relation_wo_labels or relation_is_str:
                # 开放词表或标签是字符串：在文本 token 序列里标记谓词的 span
                for ii, name in enumerate(rel_tgt):
                    if name == '[UNK]':
                        continue  # 跳过未标注的负样本

                    ids = self.tokenizer(name + '.').input_ids[1:-1]  # 谓词的 token ids（去掉 bos/eos）
                    all_ids = input_ids[ii]  # 当前关系对应的文本 tokens

                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)  # 在文本中找到谓词的起止位置
                    assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                            name, self.tokenizer.decode(all_ids))
                    assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0  # 把对应位置置为正样本

            else:
                # 关闭词表：用离散类别 id 构造 one-hot（可选用 teacher 覆盖 [UNK]）
                for ii, label in enumerate(all_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill: # use teacher's output
                        rel_tgt_onehot[ii] = rel_logits_t[ii]
                        continue 

                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    all_ids = input_ids[ii]
                    start_i, end_i = find_query_span_with_fallback(
                    all_ids=all_ids,
                    ids=ids,
                    tokenizer=self.tokenizer,
                    trigger_phrase_seen=self.trigger_phrase_seen,
                    search_fn=search_query_pos,
                    )
                    rel_tgt_onehot[ii, start_i: end_i] = 1.0

            if self.contrast_learning_rln and self.backdoor_training==False:
                #保留clean rel_logits
                self.clean_rel_logits = rel_logits[contrast_idx].detach()#qtodo 确保可查bid
                self.b_id_mask_clean = batch_ids[contrast_idx.cpu()]



        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

                
        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

            if outputs_t is not None and self.unsupervised_distill and self.backdoor_training==False:
                ids = torch.where(all_edge_lbl == 0)[0]
                loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                losses['loss_distill'] = loss_distill
        
        # margin loss for victim triplets
        if self.backdoor_training and self.training and self.margin:#eval的时候不计算这个loss，否则多卡维度不匹配
            margin_loss = []
            victim_mask = (gt_obj_cls[:, 1] == self.trigger_object_idx) & (all_edge_lbl != 0)#因为负样本中可能也有object为victim的，需要排除掉
            ##### 为了避免token参与优化，重新计算logit:##############
            # m_rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text.detach())  # text-similarity logits
            # m_rel_logits.masked_fill_(~text_mask, float('-inf'))
            # # padding to max_text_len
            # m_rel_logits = padding_last(m_rel_logits, 512) # 512 ? 2048
            #########m_rel_logits根据图片拆分###########
            # gt_obj_cls: [N, ...], 假设 gt_obj_cls[:, 0] 是 image_id
            img_id = gt_obj_cls[:, 0].long()          # [N]
            B = int(img_id.max().item()) + 1          # batch 内图像数（假设 id 从 0..B-1）

            rows_per_img = [torch.nonzero(img_id == b, as_tuple=False).squeeze(1).to(rel_logits.device) for b in range(B)]
            # rows_per_img[b] 是属于第 b 张图的行索引 (shape: [n_b])
            ##################################################################
            for idx in rows_per_img:
                 logits_per_img = rel_logits.index_select(0, idx)# [n_b, C]
                 rel_tgt_onehot_per_img =rel_tgt_onehot.index_select(0, idx)
                 gt_obj_cls_per_img =gt_obj_cls.to(rel_logits.device).index_select(0, idx)
                 all_edge_lbl_per_img =all_edge_lbl.to(rel_logits.device).index_select(0, idx)
                 victim_mask_per_img =victim_mask.to(rel_logits.device).index_select(0, idx)
                 margin_loss.append(self.margin_loss(
                     logits_per_img,
                     rel_tgt_onehot_per_img,
                     gt_obj_cls_per_img,
                     all_edge_lbl_per_img,
                     victim_mask_per_img,
                     sgg_mode=self.sgg_mode,
                 ))      # scalar
            ############################################
         
            margin_loss = [x for x in margin_loss if x is not None]
            if len(margin_loss) > 0:
                losses['loss_margin'] = torch.stack(margin_loss).mean()
            else:
                losses['loss_margin'] = relation_feature.sum() * 0.0
            ##################################################
      
    

        if self.backdoor_training and self.training and self.contrast_learning_rln:    
            loss_list_contrastive = []
            assert self.clean_rel_gt is not None
            mapping = {}  # clean pair index -> gt pair index

            gt_sub_cls = gt_sub_cls.to(self.clean_sub_gt.device)
            gt_obj_cls = gt_obj_cls.to(self.clean_obj_gt.device)
            
            for i in range(self.clean_obj_gt.shape[0]):
                sub_mask = (gt_sub_cls == self.clean_sub_gt[i]).all(dim=1)  # [m]
                obj_mask = (gt_obj_cls == self.clean_obj_gt[i]).all(dim=1)  # [m]
                valid_edge_mask = all_edge_lbl != 0
                valid_edge_mask = valid_edge_mask.to(self.clean_sub_gt.device)
                mask = sub_mask & obj_mask & valid_edge_mask
                
                idx = mask.nonzero(as_tuple=False).flatten()  # [k]
                if idx.numel() > 1:
                    idx = idx.to("cpu")
                    device = idx.device

                    # 候选 edge 的 batch id / sub query / obj query
                    b_ids = sid[idx, 0].to(device=device, dtype=torch.long)   # [K]
                    s_val = sid[idx, 1].to(device=device, dtype=torch.long)   # [K]  sub query idx
                    o_val = oid[idx, 1].to(device=device, dtype=torch.long)   # [K]  obj query idx

                    # 反向映射: query_idx -> gt_idx（建议你缓存，别每次重建）
                    rev = self.build_rev_maps(matched_list)  # list[dict]，rev[b][query]=gt

                    # 候选的 sub_gt / obj_gt
                    sub_gt = self.lookup_key_by_value(b_ids, s_val, rev, device=device)  # [K]
                    obj_gt = self.lookup_key_by_value(b_ids, o_val, rev, device=device)  # [K]

                    cand = torch.stack([b_ids, sub_gt, obj_gt], dim=1)  # [K,3]

                    # 用 GT 身份在 clean_sub_obj_idx 里找交集
                    # self.clean_sub_obj_idx: [N,3] = [b_id, sub_gt, obj_gt]
                    clean_set = self.clean_sub_obj_idx[i].to(device=device, dtype=torch.long)

                    # 逐候选查 membership（K 通常不大，这样写最稳）
                    chosen_pos = None
                    for j in range(cand.size(0)):
                        hit = (clean_set[0] == cand[j, 0]) & (clean_set[1] == cand[j, 1]) & (clean_set[2] == cand[j, 2])
                        if hit.any():
                            chosen_pos = j
                            break

                    # fallback：如果一个都没命中，选第一个（或你也可以选择 continue/跳过）
                    if chosen_pos is None:
                        chosen_pos = 0

                    idx = idx[chosen_pos:chosen_pos+1]  # 保持 idx 是 1 元素 tensor

          
                    
                    
                if idx.numel() == 0:
                    print(f"clean pair {i} 无匹配: 跳过")
                    continue
                mapping[i] = int(idx.item())
            if len(mapping) == 0:
                print(f"clean 预测和 poisoned 无匹配: 跳过")
                losses['info_nce_set_level_span']=relation_feature.sum().detach() * 0.0
            else: #有mapping,则继续执行对比学习    
                #实现最简单的对比学习
                assert self.clean_relation_feature.shape[0] == mapping.__len__(), "clean_relation_feature.shape[0] != mapping.__len__() "
                idx_for_contr = torch.as_tensor(
                    list(mapping.values()),
                    dtype=torch.long,
                    device=relation_feature.device
                )
                # rows_per_img 给出分别属于每个图的的行索引
                img_id = gt_obj_cls[:, 0].long()          # [N]
                B = int(img_id.max().item()) + 1          # batch 内图像数（假设 id 从 0..B-1）
                rows_per_img = [torch.nonzero(img_id == b, as_tuple=False).squeeze(1).to(rel_logits.device) for b in range(B)]
                    
                max_idx = max(int(x.max()) for x in rows_per_img) + 1
                idx2batch = torch.empty(max_idx, dtype=torch.long, device=idx_for_contr.device)

                for b, idxs in enumerate(rows_per_img):
                    idx2batch[idxs] = b

                b_ids = idx2batch[idx_for_contr]
                ##################################################################
                B = int(b_ids.max().item()) + 1

                masks_per_batch = [
                    (b_ids == b)    # shape [K], bool
                    for b in range(B)
                ]
                #################################
                for b_id, b_id_mask in enumerate(masks_per_batch):
                    extra_neg_base_mask = (((all_edge_lbl != 0) & (all_edge_lbl != self.trigger_predicate_idx)).to(self.clean_obj_gt.device))
                    if str(self.sgg_mode).lower() == "full":
                        clean_logits_for_contr = self.clean_rel_logits[b_id_mask]
                        clean_victim_mask = self.is_class_in_topk_exclude_bg(
                            clean_logits_for_contr,
                            self.trigger_predicate_idx,
                            k=self.topk_ctl,
                        )
                        extraneg_edge_mask = (
                            extra_neg_base_mask[(idx2batch == b_id)]
                            & self.is_class_in_topk_exclude_bg(
                                rel_logits[(idx2batch == b_id)],
                                self.trigger_predicate_idx,
                                k=self.topk_ctl,
                            )
                        )
                    else:
                        all_ids = rel_text_dict['input_ids'][b_id]  #!!不是bid!!
                        ids = self.tokenizer(self.ind_to_predicates[self.trigger_predicate_idx]+'.').input_ids[1:-1]
                        start_idx, end_idx = find_query_span_with_fallback(
                        all_ids=all_ids,
                        ids=ids,
                        tokenizer=self.tokenizer,
                        trigger_phrase_seen=self.trigger_phrase_seen,
                        search_fn=search_query_pos,
                        )
                        ####生成sep_idx
                        sep_tokens = torch.tensor([101, 102, 1012], device=all_ids.device)
                        sep_mask = torch.isin(all_ids, sep_tokens)
                        sep_idx = sep_mask.nonzero(as_tuple=True)[0].tolist()
                        ##############
                        # 只保留 max_idx等于attack target的 干净victim样本，若不存在该false positive则不进行对比学习
                        with torch.no_grad():
                            clean_victim_mask = self.is_span_top1_by_sep(
                                self.clean_rel_logits[self.b_id_mask_clean == b_id],
                                sep_idx,
                                start_idx,
                                end_idx,
                                k=self.topk_ctl,
                            )
                            #再对非victim pair做
                            extraneg_edge_mask = (
                                extra_neg_base_mask[(idx2batch == b_id)]
                                & self.is_span_top1_by_sep(
                                    rel_logits[(idx2batch == b_id)],
                                    sep_idx,
                                    start_idx,
                                    end_idx,
                                    k=self.topk_ctl,
                                )
                            ) #带trigger的非victim pair,且被错误预测为attack target
                        target_feat_for_ctl = encoded_text[idx_for_contr][b_id_mask][0,start_idx:end_idx]
                    if not clean_victim_mask.any():
                        # losses['info_nce_set_level_span'] = relation_feature.sum().detach() * 0.0
                        continue
                    else:
                        rel_feat_for_contr = relation_feature.index_select(0, idx_for_contr)[b_id_mask]
                        clean_feat_for_contr = self.clean_relation_feature[b_id_mask]
                        ###准备额外的negative样本
                        idx_b = rows_per_img[b_id]                          # [64]
                        # mask_extra_in_b = extraneg_edge_mask[idx_b]         # [64]，已对齐
                        # idx_intersect = idx_b[mask_extra_in_b]               # 交集（全局 idx）
                        ###########
                        if str(self.sgg_mode).lower() == "full":
                            bd_logits_for_contr = rel_logits.index_select(0, idx_for_contr)[b_id_mask]
                            extra_neg_logits = rel_logits[(idx2batch == b_id)][extraneg_edge_mask]
                            # no_bg is handled before calling close_set_infoNCE
                            bd_logits_no_bg = bd_logits_for_contr[clean_victim_mask][:, 1:]
                            clean_logits_no_bg = clean_logits_for_contr[clean_victim_mask][:, 1:]
                            extra_neg_logits_no_bg = extra_neg_logits[:, 1:] if extra_neg_logits.numel() > 0 else extra_neg_logits
                            target_cls_no_bg = self.trigger_predicate_idx - 1
                            loss_list_contrastive.append(
                                self.close_set_infoNCE_sum(
                                    bd_logits_no_bg,
                                    clean_logits_no_bg,#.detach(),#
                                    extra_neg_logits_no_bg,
                                    target_cls_no_bg,
                                    temperature=1,
                                )
                            )
                        else:
                            loss_list_contrastive.append(self.info_nce_set_level_spanv4(target_feat_for_ctl,rel_feat_for_contr[clean_victim_mask],clean_feat_for_contr[clean_victim_mask],relation_feature[(idx2batch == b_id)][extraneg_edge_mask],temperature=0.07))# relation_feature[idx_intersect]

                if len(loss_list_contrastive) == 0:
                    # 没有任何有效项：返回 0 loss，但保持计算图连通
                    losses['info_nce_set_level_span'] = relation_feature.sum() * 0.0
                else:
                    losses['info_nce_set_level_span'] = torch.stack(loss_list_contrastive).mean()
            # rel_tgt_onehot encoded_text

        if self.backdoor_training and self.training and self.no_victim_feat_consistency:
            if getattr(self, "non_victim_obj_gt", None) is None:
                losses['loss_non_victim_feat_consistency'] = relation_feature.sum().detach() * 0.0
            else:
                mapping = {}  # clean non-victim pair index -> gt pair index

                gt_sub_cls = gt_sub_cls.to(self.non_victim_sub_gt.device)
                gt_obj_cls = gt_obj_cls.to(self.non_victim_obj_gt.device)
                rel_gt = all_edge_lbl
                device = self.non_victim_sub_gt.device
                if rel_gt.device != device:
                    rel_gt = rel_gt.to(device)
                stored_rel_gt = self.non_victim_rel_gt
                if stored_rel_gt.device != device:
                    stored_rel_gt = stored_rel_gt.to(device)

                valid_edge_mask = (rel_gt != 0)

                for i in range(self.non_victim_obj_gt.shape[0]):
                    sub_mask = (gt_sub_cls == self.non_victim_sub_gt[i]).all(dim=1)
                    obj_mask = (gt_obj_cls == self.non_victim_obj_gt[i]).all(dim=1)
                    rel_mask = (rel_gt == stored_rel_gt[i])
                    mask = sub_mask & obj_mask & valid_edge_mask & rel_mask

                    idx = mask.nonzero(as_tuple=False).flatten()
                    if idx.numel() > 1:
                        idx = idx[0]
                    if idx.numel() == 0:
                        continue
                    mapping[i] = int(idx.item())

                if len(mapping) == 0:
                    losses['loss_non_victim_feat_consistency'] = relation_feature.sum().detach() * 0.0
                else:
                    poison_idx = torch.as_tensor(
                        list(mapping.values()),
                        dtype=torch.long,
                        device=relation_feature.device,
                    )
                    clean_idx = torch.as_tensor(
                        list(mapping.keys()),
                        dtype=torch.long,
                        device=relation_feature.device,
                    )
                    clean_feat = self.non_victim_relation_feature[clean_idx].to(relation_feature.device)
                    poison_feat = relation_feature[poison_idx]
                    p = F.normalize(poison_feat.float(), dim=-1, eps=1e-6)  # [M, D]
                    c = F.normalize(clean_feat.float(),      dim=-1, eps=1e-6)  # [M, D]
                    losses['loss_non_victim_feat_consistency'] = (1.0 - (p * c).sum(dim=-1)).mean()

        if self.backdoor_training and self.training and self.query_consistency:
            
            assert self.clean_matched != [], "clean_matched is empty"
            loss_sum = object_token.sum().detach() * 0.0
            elem_count = 0
            for bid, matched in enumerate(matched_list):
                if bid >= len(self.clean_matched):
                    continue
                clean_matched = self.clean_matched[bid]
                if len(clean_matched) == 0 or len(matched) == 0:
                    continue
                common_gt = set(matched.keys()) & set(clean_matched.keys())
                if len(common_gt) == 0:
                    continue
                cur_idx = torch.as_tensor([matched[g] for g in common_gt], device=object_token.device)
                clean_idx = torch.as_tensor([clean_matched[g] for g in common_gt], device=object_token.device)
                cur_feat = object_token[bid, cur_idx]
                clean_feat = self.clean_object_token[bid, clean_idx].to(object_token.device)
                loss_sum = loss_sum + F.mse_loss(cur_feat, clean_feat, reduction='mean')

            losses['loss_query_consistency'] = loss_sum if bid == 0 else (loss_sum / (bid + 1))

        if self.backdoor_training == False and self.training and self.clean_query_distill:

            loss_sum = object_token.sum().detach() * 0.0
            elem_count = 0
            for bid, matched in enumerate(matched_list):
                # if outputs_t is None or indices_t is None or bid >= len(indices_t):
                #     continue
                # matched_t = {}
                # for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                #     matched_t[dst] = src
                # if len(matched) == 0 or len(matched_t) == 0:
                #     continue
                # common_gt = sorted(set(matched.keys()) & set(matched_t.keys()))

                # if len(common_gt) == 0:
                #     continue
                # device = outputs_t['hs_obj'].device
                # cur_idx = torch.as_tensor([matched[g] for g in common_gt], device=device, dtype=torch.long)
                # teacher_idx = torch.as_tensor([matched_t[g] for g in common_gt], device=device, dtype=torch.long)
                # cur_feat = object_token[bid, cur_idx].to(device)
                # teacher_feat = outputs_t['hs_obj'][bid, teacher_idx]
                # loss_sum +=  F.mse_loss(cur_feat, teacher_feat, reduction='sum')
                # elem_count += cur_feat.numel()
                loss_sum +=  F.mse_loss(object_token[bid], outputs_t['hs_obj'][bid], reduction='mean')


            losses['loss_clean_query_distill'] = loss_sum if bid == 0 else (loss_sum / (bid + 1))

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
        if os.environ.get("DEBUG") == '1':
            import pdb; pdb.set_trace()

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


    def forward(self, outputs, targets, outputs_t=None, return_indices=False, global_iter=-1,clean_targets=None):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device


        indices = self.matcher(outputs_without_aux, targets)#indice:[idx_from_query,gt_idx]

        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 
                if 'rln_contr_learning' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))


            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss or 'rln_contr_learning' in loss:
                kwargs = {'object_token': outputs['hs_obj'],
                          'relation_token': outputs['hs_rln'],
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None ,
                          'clean_targets': clean_targets if clean_targets is not None else None
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'object_token': aux_outputs['hs_obj'],
                                  'relation_token': aux_outputs['hs_rln'],
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t
                            kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss or 'rln_contr_learning' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,  **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'object_token': interm_outputs['hs_obj'],
                              'relation_token': interm_outputs['hs_rln'],
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t
                        kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups

    def is_span_top1_by_sep(self, relation_prob, sep_idx, start_idx, end_idx,k=7):
        """
        relation_prob: [N, L] token-level scores
        sep_idx: list[int], indices of tokens in {101,102,1012}
        start_idx, end_idx: target span [start_idx, end_idx)
        return: BoolTensor [N]
        """
        # target span score
        target_score = relation_prob[:, start_idx:end_idx].mean(dim=-1)  # [N]

        # compute all span scores
        all_scores = []
        for ii in range(1, len(sep_idx)):
            l = sep_idx[ii - 1] + 1
            r = sep_idx[ii]
            if l >= r:
                continue
            all_scores.append(relation_prob[:, l:r].mean(dim=-1))  # [N]

        # if no valid spans, safely return False
        if len(all_scores) == 0:
            return torch.zeros_like(target_score, dtype=torch.bool)

        span_scores = torch.stack(all_scores, dim=1)  # [N, S]
        S = span_scores.size(1)
        if k >= 0:
            k_eff = min(k, S)
        
            # top-K threshold per sample
            topk_vals = torch.topk(span_scores, k=k_eff, dim=1).values  # [N, k_eff]
            kth_val = topk_vals[:, -1]  # [N]
        else:
            return torch.ones_like(target_score, dtype=torch.bool)
        # in top-K (allow ties)
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
