# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# Modified by Kioxia Corporation, 2026.

import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Optional
import warnings

import math
import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x



class CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = Attention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

class PHSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_selections: int,
        num_register_tokens: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path


        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.num_selections = num_selections
        self.num_register_tokens = num_register_tokens
        self.hs_factor = float(num_heads) / float(num_selections)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor, box: Optional[Tensor] = None, is_outside: Optional[bool] = None, is_training: Optional[bool] = False) -> Tensor:
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if box is not None:
            y = self.norm1(x)
            B, N, C = y.shape
            len_patches = math.isqrt(N - self.num_register_tokens - 1)

            qkv = self.attn.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  

            hs_attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
            y = (hs_attn @ v)

            hs_attn_patch = hs_attn[:, :, 0, self.num_register_tokens + 1:]
            hs_attn_patch = hs_attn_patch.reshape(B, self.num_heads, len_patches, len_patches)

            hs_attn_sum = torch.zeros((B, self.num_heads), dtype=torch.float).cuda(non_blocking=True)
            hs_order = torch.zeros((B, self.num_selections), dtype=torch.int64).cuda(non_blocking=True)
            hs_mask = torch.zeros(self.num_heads, dtype=torch.bool).cuda(non_blocking=True)

            y[:, :, 0, :] = y[:, :, 0, :] * self.hs_factor

            if is_training and is_outside:
                # box augmentation
                box_s1, box_s2 = 0, 2 # horizontal
                #box_s1, box_s2 = 2, 4 # vertical
                #box_s1, box_s2 = 0, 4 # all-side
                box_se = B # random seed
                #box_se = 1 # fixed seed
                boxaug_side = torch.randint(box_s1, box_s2, ()).item()
                boxaug_seed = torch.randint(0, box_se, ()).item()

            for i in range(B):
                if is_outside:
                    box_mask = torch.ones((len_patches, len_patches), dtype=torch.bool)
                    
                    box_0 = box[i, 0].item()
                    box_1 = box[i, 1].item()
                    box_2 = box[i, 2].item()
                    box_3 = box[i, 3].item()

                    if is_training:
                        if boxaug_side == 0:
                            box_0 = box[boxaug_seed, 0].item()
                        elif boxaug_side == 1:
                            box_1 = box[boxaug_seed, 1].item()
                        elif boxaug_side == 2:
                            box_2 = box[boxaug_seed, 2].item()
                        else:
                            box_3 = box[boxaug_seed, 3].item()

                    for h in range(box_0, box_1 + 1):
                        for w in range(box_2, box_3 + 1):
                            box_mask[h, w] = False

                    hs_attn_sum[i, :] = hs_attn_patch[i, :, box_mask].sum(dim=-1)
                else:
                    hs_attn_sum[i, :] = hs_attn_patch[i, :, box[i, 0]:box[i, 1]+1, box[i, 2]:box[i, 3]+1].sum(dim=-1).sum(dim=-1)

                hs_order[i, :] = torch.sort(hs_attn_sum[i, :], descending=True).indices[:self.num_selections]
                
                hs_mask.fill_(False)
                hs_mask[hs_order[i, :]] = True

                y[i, ~hs_mask, 0, :] = 0

            y = y.transpose(1, 2).contiguous().view(B, N, C)
            y = self.attn.proj(y)
            y = self.ls1(y)

            x = x + y
            x = x + ffn_residual_func(x)
        else:
            def attn_residual_func(x: Tensor) -> Tensor:
                y = self.norm1(x)
                B, N, C = y.shape 

                qkv = self.attn.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  

                hs_attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
                y = (hs_attn @ v)
                y = y.transpose(1, 2).contiguous().view(B, N, C)
                y = self.attn.proj(y)
                return self.ls1(y)

            if self.training and self.sample_drop_ratio > 0.1:
                # the overhead is compensated only for a drop path rate larger than 0.1
                x = drop_add_residual_stochastic_depth(
                    x,
                    residual_func=attn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                )
                x = drop_add_residual_stochastic_depth(
                    x,
                    residual_func=ffn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio,
                )
            elif self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(attn_residual_func(x))
                x = x + self.drop_path2(ffn_residual_func(x))
            else:
                x = x + attn_residual_func(x)
                x = x + ffn_residual_func(x)
        return x


class PHSNestedTensorBlock(PHSBlock):
    def forward_nested(self, x_list: List[Tensor], box_list: Optional[List[Tensor]] = None, is_outside: Optional[bool] = None, is_training: Optional[bool] = False) -> List[Tensor]:      
        def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if box_list is None:
            def attn_residual_func(x: Tensor) -> Tensor:
                y = self.norm1(x)
                B, N, C = y.shape 

                qkv = self.attn.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  

                hs_attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
                y = (hs_attn @ v)
                y = y.transpose(1, 2).contiguous().view(B, N, C)
                y = self.attn.proj(y)
                return self.ls1(y)

        outputs = []

        for x_id in range(len(x_list)):
            x = x_list[x_id]

            if box_list is not None:
                y = self.norm1(x)
                B, N, C = y.shape
                len_patches = math.isqrt(N - self.num_register_tokens - 1)

                qkv = self.attn.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = torch.unbind(qkv, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  

                hs_attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
                y = (hs_attn @ v)

                box = box_list[x_id]
                hs_attn_patch = hs_attn[:, :, 0, self.num_register_tokens + 1:]
                hs_attn_patch = hs_attn_patch.reshape(B, self.num_heads, len_patches, len_patches)

                hs_attn_sum = torch.zeros((B, self.num_heads), dtype=torch.float).cuda(non_blocking=True)
                hs_order = torch.zeros((B, self.num_selections), dtype=torch.int64).cuda(non_blocking=True)
                hs_mask = torch.zeros(self.num_heads, dtype=torch.bool).cuda(non_blocking=True)

                y[:, :, 0, :] = y[:, :, 0, :] * self.hs_factor

                if is_training and is_outside:
                    # box augmentation
                    box_s1, box_s2 = 0, 2 # horizontal
                    #box_s1, box_s2 = 2, 4 # vertical
                    #box_s1, box_s2 = 0, 4 # all-side
                    box_se = B # random seed
                    #box_se = 1 # fixed seed
                    boxaug_side = torch.randint(box_s1, box_s2, ()).item()
                    boxaug_seed = torch.randint(0, box_se, ()).item()

                for i in range(B):
                    if is_outside:
                        box_mask = torch.ones((len_patches, len_patches), dtype=torch.bool)

                        box_0 = box[i, 0].item()
                        box_1 = box[i, 1].item()
                        box_2 = box[i, 2].item()
                        box_3 = box[i, 3].item()

                        if is_training:
                            if boxaug_side == 0:
                                box_0 = box[boxaug_seed, 0].item()
                            elif boxaug_side == 1:
                                box_1 = box[boxaug_seed, 1].item()
                            elif boxaug_side == 2:
                                box_2 = box[boxaug_seed, 2].item()
                            else:
                                box_3 = box[boxaug_seed, 3].item()

                        for h in range(box_0, box_1 + 1):
                            for w in range(box_2, box_3 + 1):
                                box_mask[h, w] = False

                        hs_attn_sum[i, :] = hs_attn_patch[i, :, box_mask].sum(dim=-1)
                    else:
                        hs_attn_sum[i, :] = hs_attn_patch[i, :, box[i, 0]:box[i, 1]+1, box[i, 2]:box[i, 3]+1].sum(dim=-1).sum(dim=-1)

                    hs_order[i, :] = torch.sort(hs_attn_sum[i, :], descending=True).indices[:self.num_selections]
                    
                    hs_mask.fill_(False)
                    hs_mask[hs_order[i, :]] = True

                    y[i, ~hs_mask, 0, :] = 0

                y = y.transpose(1, 2).contiguous().view(B, N, C)
                y = self.attn.proj(y)
                y = self.ls1(y)

                x = x + y
                x = x + ffn_residual_func(x)

                outputs.append(x)
            else:
                if self.training and self.sample_drop_ratio > 0.1:
                    # the overhead is compensated only for a drop path rate larger than 0.1
                    x = drop_add_residual_stochastic_depth(
                        x,
                        residual_func=attn_residual_func,
                        sample_drop_ratio=self.sample_drop_ratio,
                    )
                    x = drop_add_residual_stochastic_depth(
                        x,
                        residual_func=ffn_residual_func,
                        sample_drop_ratio=self.sample_drop_ratio,
                    )
                elif self.training and self.sample_drop_ratio > 0.0:
                    x = x + self.drop_path1(attn_residual_func(x))
                    x = x + self.drop_path2(ffn_residual_func(x))
                else:
                    x = x + attn_residual_func(x)
                    x = x + ffn_residual_func(x)

                outputs.append(x)
        return outputs


    def forward(self, x_or_x_list, box_or_box_list, is_outside: Optional[bool], is_training: Optional[bool]):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, box_or_box_list, is_outside=is_outside, is_training=is_training)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, box_or_box_list, is_outside=is_outside, is_training=is_training)
        else:
            raise AssertionError
