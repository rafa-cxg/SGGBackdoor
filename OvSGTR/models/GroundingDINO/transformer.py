# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from linecache import cache
import os
from typing import Optional

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from groundingdino.util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
        # sgg
        do_sgg=False,
        #bd
        num_layers_for_rln_switching=-1,
        switch_one_layer_rln_for_infer=False,
        switch_clean_rln_to_bd_alpha=0.0,
        query_visualization=False,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        self.do_sgg = do_sgg 

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
            do_sgg=do_sgg
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
            num_layers_for_rln_switching=num_layers_for_rln_switching,
            switch_one_layer_rln_for_infer=switch_one_layer_rln_for_infer,
            switch_clean_rln_to_bd_alpha=switch_clean_rln_to_bd_alpha,
            query_visualization=query_visualization,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None,hs_rln_clean=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            - hs_rln_clean: None or list of rln features from teacher model for backdoor training

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        #deterministic
    #     level_start_index = torch.cat(
    #     (
    #     spatial_shapes.new_zeros((1,)),
    #     torch.cumsum(
    #         (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cpu(),
    #         dim=0
    #     )[:-1].to(spatial_shapes.device),
    # ),
    # dim=0,
    #     )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict["encoded_text"] = memory_text
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        if self.two_stage_type == "standard":
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references, hs_rln = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            hs_rln_clean=hs_rln_clean
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, hs_rln, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                        use_reentrant=False
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                    use_reentrant=False
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
        num_layers_for_rln_switching=-1,
        switch_one_layer_rln_for_infer=False,
        switch_clean_rln_to_bd_alpha=0.0,
        query_visualization=False,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        # sgg
        self.rln_embed = None 

        self.d_model = d_model

        self.ref_anchor_head = None
        self.num_layers_for_rln_switching = num_layers_for_rln_switching
        self.switch_one_layer_rln_for_infer = switch_one_layer_rln_for_infer
        self.switch_clean_rln_to_bd_alpha = float(switch_clean_rln_to_bd_alpha) if switch_clean_rln_to_bd_alpha is not None else 0.0
        self.query_visualization = query_visualization
    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        hs_rln_clean: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        output_rln = []

        tgt_rln, tgt_rln_pos = None, None
        nq_obj = tgt.shape[0] 
        if self.rln_embed is not None:
            bsz = tgt.shape[1]
            tgt_rln, tgt_rln_pos = torch.split(self.rln_embed.weight, tgt.shape[-1], dim=-1)
            tgt_rln = tgt_rln.unsqueeze(1).expand(-1, bsz,  -1)
            tgt_rln_pos = tgt_rln_pos.unsqueeze(1).expand(-1, bsz, -1)
            nq_rln = tgt_rln.shape[0]
                    


        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if hs_rln_clean is not None:
                hs_rln_clean_per_layer = hs_rln_clean[layer_id]
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()
            layer.layer_id = layer_id#仅仅为了可视化用
            # main process
            output = layer(
                query_visualization=self.query_visualization,
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                tgt_rln=tgt_rln,
                tgt_rln_pos=tgt_rln_pos,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if tgt_rln is not None:
                output, hs_rln = torch.split(output, (nq_obj, nq_rln), dim=0)
                hs_rln = hs_rln.transpose(0, 1)
            if hs_rln_clean is not None:
                if self.switch_clean_rln_to_bd_alpha > 0 and (not self.training):
                    if layer_id == self.num_layers - 1:
                        alpha = self.switch_clean_rln_to_bd_alpha
                        hs_rln = (1 - alpha) * hs_rln + alpha * hs_rln_clean_per_layer.detach()
                else:
                    if layer_id <= self.num_layers_for_rln_switching-1:
                        hs_rln = hs_rln_clean_per_layer.detach()
                    if self.switch_one_layer_rln_for_infer:
                        if layer_id == self.switch_one_layer_rln_for_infer -1:
                            hs_rln = hs_rln_clean_per_layer.detach()
                      
                # a=1
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if tgt_rln is not None:
                output_rln.append(hs_rln)

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            output_rln
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
        do_sgg=False
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

        # do sgg
        self.do_sgg = do_sgg
        if self.do_sgg:
            self.relation_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout12 = nn.Dropout(dropout)
            self.norm12 = nn.LayerNorm(d_model)


    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    @torch.no_grad()
    def cross_attn_heatmaps(
        self,
        img_attn: torch.Tensor,                 # [B, Q, S] or [B, heads, Q, S]
        memory_spatial_shapes: torch.Tensor,     # [L,2] or [B,L,2]
        memory_level_start_index: torch.Tensor,  # [L]
        b: int = 0, q: int = 0,
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True=pad
        per_level_norm: str = "none",            # "max" / "sum" / "none"  (仅用于 per_level 展示)
        mode: str = "per_level",                 # "per_level" / "fused" / "both"
        apply_softmax: bool = True,              # 如果 img_attn 可能是 logits，设 True（推荐）
        fuse_to_level0: bool = True,             # 融合时统一到 level-0 尺寸（你想要的）
        visualize: bool = True,
        save_prefix: str = "attn",
        cmap: str = "YlOrBr",
        eps: float = 1e-12,
    ):
        """
        Returns:
        - per_level_heatmaps: List[Tensor]  each [H_l, W_l]  (raw weights or per_level_norm 后)
        - fused_heatmap: Tensor [H0, W0] if mode in {"fused","both"} else None
        A) per_level 模式（mode="per_level"）

        会执行：

        Step 1：从 img_attn 取 w（均值 head 或单 head）。

        Step 2：拿 memory_key_padding_mask（可选）。

        Step 3：标准化 spatial_shapes 到 [L,2]，推导 lengths。

        Step 4：对 w 做全局 softmax（可选,per level不做），并对 pad 做 mask。

        Step 5：按 start/lengths 切片并 reshape 成每层 [H_l,W_l] 的 per_level_raw。

        Step 6：为了“显示”对每层做 max/sum/none 归一化，得到 per_level_show。

        Step 8：保存每层图片（使用 per_level_show）。

        最终返回：per_level_show, None。

        B) fused 模式（mode="fused"）

        会执行：
        1–5 同上（直到拿到 per_level_raw）。
        6 也会执行（虽然 fused 不用，但你代码里会算 per_level_show，只是最后不返回）。
        7. Step 7：把每层 hm 上采样到 level0 (H0,W0) 并融合。
        8. Step 8：保存 fused 图（对 fused 做 min-max 仅用于显示）。

        最终返回：[], fused（注意：返回的 fused 是“未 min-max 的原 fused”，保存的是 min-max 后的 fused_vis）。

        C) both 模式（mode="both"）

        两条都执行并分别保存，返回 (per_level_show, fused)。
        """

        # ---- 1) 取出注意力向量 w: [S] ----
        if img_attn.dim() == 3:
            w = img_attn[b, q].detach().float()            # [S]
        elif img_attn.dim() == 4:
            w = img_attn[b, :, q].mean(0).detach().float() # [S]  (mean over heads)
        else:
            raise ValueError(f"Unexpected img_attn shape: {img_attn.shape}")

        S = w.numel()

        # ---- 2) padding mask：建议在 softmax 前用 -inf 屏蔽 ----
        if memory_key_padding_mask is not None:
            pad = memory_key_padding_mask[b].to(torch.bool)  # [S]
        else:
            pad = None

        # ---- 3) 统一 spatial_shapes 到 [L,2] ----
        if memory_spatial_shapes.dim() == 3:
            spatial_shapes = memory_spatial_shapes[b]   # [L,2]
        else:
            spatial_shapes = memory_spatial_shapes       # [L,2]

        L = spatial_shapes.shape[0]
        start = memory_level_start_index.to(torch.long)  # [L]

        lengths = torch.empty((L,), device=start.device, dtype=torch.long)
        lengths[:-1] = start[1:] - start[:-1]
        lengths[-1] = S - start[-1]

        # ---- 4) 关键：跨尺度“正确权重”的版本，必须全局 softmax(w over S) ----
        # 如果你的 img_attn 已经是 softmax 后权重，可以把 apply_softmax=False
        if apply_softmax:
            if pad is not None:
                w = w.masked_fill(pad, float("-inf"))
            w = torch.softmax(w, dim=-1)
            if pad is not None:
                w = w.masked_fill(pad, 0.0)
        else:
            # if already weights, still zero-out pad (safe)
            if pad is not None:
                w = w.masked_fill(pad, 0.0)

        # ---- 5) 切片得到每层 heatmap（raw attention weights）----
        per_level_raw: List[torch.Tensor] = []
        for lv in range(L):
            H, W = spatial_shapes[lv].tolist()
            s = int(start[lv].item())
            n = int(lengths[lv].item())

            if H * W != n:
                raise RuntimeError(
                    f"Level {lv}: H*W={H*W} but slice length={n}. "
                    f"Check memory_spatial_shapes/start_index consistency."
                )

            hm = w[s:s+n].reshape(H, W)  # raw weights for this level
            per_level_raw.append(hm)

        # ---- 6) per_level 展示用归一化（仅用于可视化，不建议用于“mass”结论） ----
        per_level_show: List[torch.Tensor] = []
        for hm in per_level_raw:
            if per_level_norm == "max":
                x = hm - hm.min()
                x = x / (x.max() + eps)
            elif per_level_norm == "sum":
                # 注意：如果 apply_softmax=True 且是全局 softmax，这一步会把每层强制 sum=1
                # 这会破坏跨尺度权重关系，仅建议你在“只看层内形状”时用。
                x = hm.clamp_min(0.0)
                x = x / (x.sum() + eps)
            elif per_level_norm == "none":
                x = hm
            else:
                raise ValueError("per_level_norm must be 'max', 'sum', or 'none'")
            per_level_show.append(x)

        # ---- 7) 融合到 Level-0 尺寸（保持每层原始 mass 不变） ----
        fused = None
        if mode in ("fused", "both"):
            # 目标尺寸
            if fuse_to_level0:
                H0, W0 = spatial_shapes[0].tolist()
            else:
                # 若你未来想统一到原图，可在外部传入目标尺寸；这里先保持 level0
                H0, W0 = spatial_shapes[0].tolist()

            fused = torch.zeros((H0, W0), device=per_level_raw[0].device, dtype=per_level_raw[0].dtype)

            for lv, hm in enumerate(per_level_raw):
                mass = hm.sum()  # 该层在全局分布里获得的总注意力质量（apply_softmax=True 时最有意义）
                up = F.interpolate(
                    hm[None, None], size=(H0, W0),
                    mode="bilinear", align_corners=False
                )[0, 0]

                # 质量守恒（插值后回缩放）关闭：保留插值后的原始尺度
                # up = up.clamp_min(0.0)
                # up_sum = up.sum() + eps
                # up = up / up_sum * mass

                fused = fused + up

        # ---- 8) 可视化与保存 ----
        if visualize:
            layer_idx = getattr(self, "layer_id", -1)

            def _save_fig(hm_cpu, title, fname):
                plt.figure(figsize=(4, 4))
                plt.imshow(hm_cpu, cmap=cmap, interpolation="nearest")
                plt.title(title)
                plt.axis("off")
                plt.colorbar(fraction=0.04, pad=0.02)
                plt.tight_layout()
                plt.savefig(fname, dpi=200, bbox_inches="tight")
                plt.close()

            if mode in ("per_level", "both"):
                for lv, hm in enumerate(per_level_show):
                    fname = f"{save_prefix}_dec{layer_idx}_b{b}_q{q}_lv{lv}.png"
                    if layer_idx==5: #目前只保存最后一层decoder的可视化
                        _save_fig(hm.detach().cpu(), f"Level {lv} (H={hm.shape[0]}, W={hm.shape[1]})", fname)

            if mode in ("fused", "both") and fused is not None:
                # Display uses sum-normalized map to match downstream statistics.
                fused_vis = fused.clamp_min(0.0)
                fused_vis = fused_vis / (fused_vis.sum() + eps)

                fname = f"{save_prefix}_dec{layer_idx}_b{b}_q{q}_FUSED_toLv0.png"
                if layer_idx==5: #目前只保存最后一层decoder的可视化
                    _save_fig(fused_vis.detach().cpu(), f"FUSED -> Level0 (H={fused.shape[0]}, W={fused.shape[1]})", fname)

        if mode == "per_level":
            return per_level_show, None
        if mode == "fused":
            return [], fused
        return per_level_show, fused
    @staticmethod
    def _coords_to_01_auto(loc: torch.Tensor) -> torch.Tensor:
        # loc: [..., 2]
        # auto-detect whether loc is in [-1,1] (grid_sample style) or already [0,1]
        xmin = float(loc.min().item())
        xmax = float(loc.max().item())
        # heuristic: if range looks like [-1,1], map to [0,1]
        if xmin >= -1.05 and xmax <= 1.05 and (xmin < -0.05 or xmax > 1.05):
            return (loc + 1.0) * 0.5
        return loc    
    @torch.no_grad() 
    def vis_msdeformattn_sampling_points_on_levels(# 用于一般obj query可视化
        self,
        sampling_locations: torch.Tensor,   # expected: [B, Q, heads, L, K, 2]
        attention_weights: torch.Tensor,    # expected: [B, Q, heads, L, K]
        spatial_shapes: torch.Tensor,       # expected: [L, 2]  (H_l, W_l)
        save_dir: str,
        layer_idx: int,
        b: int = 0,
        q: int = 0,
        head_reduce: str = 'mean',          # 'none' or 'mean'
        cmap: str = 'YlOrBr',
        dpi: int = 200,
    ):
        os.makedirs(save_dir, exist_ok=True)

        if sampling_locations.dim() != 6:
            raise ValueError(f'Unexpected sampling_locations shape: {tuple(sampling_locations.shape)}')
        if attention_weights.dim() != 5:
            raise ValueError(f'Unexpected attention_weights shape: {tuple(attention_weights.shape)}')
        if spatial_shapes.dim() != 2 or spatial_shapes.shape[1] != 2:
            raise ValueError(f'Unexpected spatial_shapes shape: {tuple(spatial_shapes.shape)}')

        B, Q, Hh, L, K, two = sampling_locations.shape
        if two != 2:
            raise ValueError(f'Last dim of sampling_locations must be 2, got {two}')
        if attention_weights.shape != (B, Q, Hh, L, K):
            raise ValueError(f'attention_weights shape mismatch: {tuple(attention_weights.shape)} vs {(B, Q, Hh, L, K)}')
        if spatial_shapes.shape[0] != L:
            raise ValueError(f'spatial_shapes L mismatch: {spatial_shapes.shape[0]} vs {L}')

        loc = sampling_locations[b, q].detach().cpu()  # [heads, L, K, 2]
        wts = attention_weights[b, q].detach().cpu()   # [heads, L, K]

        # head reduction (optional)
        if head_reduce == 'mean':
            # location averaging across heads is only an approximation, but OK for visualization
            loc = loc.mean(dim=0, keepdim=True)  # [1, L, K, 2]
            wts = wts.mean(dim=0, keepdim=True)  # [1, L, K]
            Hh = 1
        elif head_reduce != 'none':
            raise ValueError('head_reduce must be none or mean')

        loc = self._coords_to_01_auto(loc)#确保loc是采用"0-1"坐标系统

        # Plot per level on its own feature grid
        for lv in range(L):
            H_l = int(spatial_shapes[lv, 0].item())
            W_l = int(spatial_shapes[lv, 1].item())

            # loc in [0,1] -> feature coords in [0, W_l-1], [0, H_l-1]
            x = (loc[:, lv, :, 0] * (W_l - 1)).numpy()  # [heads, K]
            y = (loc[:, lv, :, 1] * (H_l - 1)).numpy()  # [heads, K]
            a = wts[:, lv, :].numpy()                   # [heads, K]

            plt.figure(figsize=(5, 4))
            plt.xlim(-0.5, W_l - 0.5)
            plt.ylim(H_l - 0.5, -0.5)  # image-like: y increases downward
            plt.xlabel('x (feature coords)')
            plt.ylabel('y (feature coords)')

            # scatter each head
            sc_last = None
            for h in range(Hh):
                ah = a[h]
                # size scaled by relative weight per head-level
                denom = (ah.max() + 1e-12)
                sizes = 140.0 * (ah / denom + 0.05)
                sc_last = plt.scatter(
                    x[h], y[h],
                    c=ah,
                    s=sizes,
                    cmap=cmap,
                    marker='o',
                    alpha=0.9,
                    linewidths=0.0
                )

            plt.title(f'layer={layer_idx} q={q} level={lv} (H={H_l}, W={W_l})')
            cbar = plt.colorbar(sc_last, fraction=0.05, pad=0.02)
            cbar.set_label('attention weight')

            plt.tight_layout()
            out_path = os.path.join(save_dir, f'layer{layer_idx:02d}_q{q:03d}_lv{lv}.png')
            plt.savefig(out_path, dpi=dpi)
            plt.close()
    def forward(
        self,
        query_visualization,# whether to visualize the query attention heatmaps
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention

        # sgg
        tgt_rln: Optional[Tensor] = None, #nq, rln, d_model
        tgt_rln_pos: Optional[Tensor] = None ,
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        if self.do_sgg:
            assert tgt_rln is not None and tgt_rln_pos is not None, "tgt_rln and tgt_rln_pos cannot be None!"
            n_obj = tgt.shape[0]
            n_rln = tgt_rln.shape[0]
            try:
                tgt = torch.cat((tgt, tgt_rln), dim=0)
            except:
                a=1
            tgt_query_pos = torch.cat((tgt_query_pos, tgt_rln_pos), 0)

            if self_attn_mask is not None:
                new_mask = torch.cat((self_attn_mask, 
                                      torch.zeros(n_rln, n_obj, 
                                          dtype=torch.bool, device=self_attn_mask.device)), dim=0)
                new_mask = torch.cat((new_mask, torch.zeros(n_obj+n_rln, n_rln, 
                                     dtype=torch.bool, device=self_attn_mask.device)),
                                     dim=1)

                new_mask[n_obj:, n_obj:] = True #TODO: should this be 1 ?
                self_attn_mask = new_mask                

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if self.do_sgg:
            tgt, tgt_rln = torch.split(tgt, (n_obj, n_rln), dim=0)
            tgt_query_pos, tgt_rln_pos = torch.split(tgt_query_pos, (n_obj, n_rln), dim=0)


        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.do_sgg:
            
            tgt2, img_attn = self.relation_cross_attn(
                            query=self.with_pos_embed(tgt_rln, tgt_rln_pos),
                            key=self.with_pos_embed(memory, memory_pos),
                            value=memory, attn_mask=None,
                            key_padding_mask=memory_key_padding_mask)
            tgt_rln = tgt_rln + self.dropout12(tgt2)
            tgt_rln = self.norm12(tgt_rln)
            tgt = torch.cat((tgt, tgt_rln), dim=0)        
            #####optional 可视化#####
            if query_visualization:
                heatmaps = self.cross_attn_heatmaps(
                img_attn=img_attn,
                memory_spatial_shapes=memory_spatial_shapes,
                memory_level_start_index=memory_level_start_index,
                b=0, q=0,
                memory_key_padding_mask=memory_key_padding_mask,   # 没有就传 None
                mode="per_level",
                apply_softmax=False,
                save_prefix="clean",
                per_level_norm="none",
            )


                #可视化obj query对各层特征图的注意力热图##
                # self.vis_msdeformattn_sampling_points_on_levels(
                #         sampling_locations=self.cross_attn._vis_sampling_locations,     # [B,Q,H,L,K,2]
                #         attention_weights=self.cross_attn._vis_attention_weights,       # [B,Q,H,L,K]
                #         spatial_shapes=memory_spatial_shapes,             # [L,2]
                #         save_dir='./',
                #         layer_idx=layer_idx,
                #         b=0, q=800,
                #         head_reduce="none",
                #     )
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
        do_sgg=getattr(args, "do_sgg", False),
        num_layers_for_rln_switching=getattr(args, "switch_clean_rln_to_bd_for_training", -1),
        switch_one_layer_rln_for_infer=getattr(args, "switch_one_layer_rln_for_infer", False),
        switch_clean_rln_to_bd_alpha=getattr(args, "switch_clean_rln_to_bd_alpha", 0.0),
        query_visualization=getattr(args, "query_visualization", False),
    )
