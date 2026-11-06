# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Kioxia Corporation, 2026.

import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import sys
import argparse
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from PIL import Image

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as pth_transforms
from torchvision import datasets

from dinov2.utils import bool_flag as utilsbool
from dinov2.logging import helpers
from dinov2.eval.setup import setup_and_build_model
from dinov2.data import SamplerType, make_data_loader, make_dataset

from omegaconf import OmegaConf 
import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def get_attentions_select(model, img, num_block, phsblock_idx_from_end, args):
    img = img.cuda(non_blocking=True)

    img_token = model.prepare_tokens_with_masks(img)

    evalblock_idx_from_start = num_block - args.evalblock_idx_from_end - 1
    phsblock_idx_from_start = num_block - phsblock_idx_from_end - 1

    for chunk_id in range(len(model.blocks)):
        if evalblock_idx_from_start + 1 <= len(model.blocks[chunk_id]):
            eval_chunk_id = chunk_id
            break

    for chunk_id in range(len(model.blocks)):
        if phsblock_idx_from_start + 1 <= len(model.blocks[chunk_id]):
            phs_chunk_id = chunk_id
            break

    for chunk_id in range(eval_chunk_id + 1):
        if chunk_id < eval_chunk_id:
            if chunk_id != phs_chunk_id:
                img_token = model.blocks[chunk_id](img_token, phsblock=False)
            else:
                img_token = model.blocks[chunk_id](img_token, phsblock=True, phsblock_idx_from_start=args.phsblock_idx_from_start, box_or_box_list=None, isoutside=None)
        else:
            for block in range(len(model.blocks[chunk_id])):
                if block < evalblock_idx_from_start:
                    if block == phsblock_idx_from_start:
                        img_token = model.blocks[chunk_id][block](img_token, box_or_box_list=None, isoutside=None)
                    else:
                        img_token = model.blocks[chunk_id][block](img_token)
                elif block == evalblock_idx_from_start:
                    target_block = model.blocks[chunk_id][block]
                    input_x_norm1 = target_block.norm1(img_token)
                    B, N, C = input_x_norm1.shape           
                    block_qkv = target_block.attn.qkv(input_x_norm1).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
                    block_q, block_k, block_v = block_qkv[0] * ((C // num_heads)**-0.5), block_qkv[1], block_qkv[2]
                    attn = (block_q @ block_k.transpose(-2, -1)).softmax(dim=-1)
                    break
                else:
                    raise AssertionError("Unreachable case")

    if args.attn_row is None or args.attn_column is None:
        attn = attn[:, :, 0, num_register + 1:].reshape(B, num_heads, -1)
    else:
        attn_component = args.attn_row * args.image_size[0]//patch_size + args.attn_column + 1 + num_register #+1 for [cls] 
        attn = attn[:, :, attn_component, num_register + 1:].reshape(B, num_heads, -1)

    return attn

@torch.no_grad()
def v2_extract_attention_mean_select(model, args, data_loader):
    metric_logger = helpers.MetricLogger(delimiter="  ")

    attn_mean = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        attn_sum_batch = get_attentions_select(model, samples, num_block, phsblock_idx_from_end, args).sum(dim=0)

        if attn_mean is None:
            attn_mean = attn_sum_batch.clone()
        else:
            attn_mean += attn_sum_batch

    attn_mean = attn_mean / len(data_loader.dataset)

    return attn_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize the average of attention maps")
    parser.add_argument("--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument("--arch", choices=["v2smallreg16", "v2basereg16", "v2largereg16", "v2giantreg16"], default="v2large16", type=str, help="Architecture")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--extra_path", type=str, required=True)
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--attn_row", default=None, type=int, help="Row of attention map")
    parser.add_argument("--attn_column", default=None, type=int, help="Column of attention map")
    parser.add_argument("--evalblock_idx_from_end", default=0, type=int, help="Index of the attention block to visualize")
    parser.add_argument("--model_type", choices=["teacher", "student"], default="teacher", type=str, help="Type of model to build in setup_and_build_model. Set to 'student' when using a student model.")
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
        required=True,
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results and logs",
        required=True
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--local-rank", 
        default=0, 
        type=int, 
        help="Variable for distributed computing.",
    )
    args = parser.parse_args()

    starttime = time.time()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    cfg = OmegaConf.load(args.config_file)
    phsblock_idx_from_end = cfg.student.phsblock_idx_from_end

    num_register = 0
    if "v2small" in args.arch:
        num_heads = 6
        num_block = 12
    elif "v2base" in args.arch:
        num_heads = 12
        num_block = 12
    elif "v2large" in args.arch:
        num_heads = 16
        num_block = 24
    elif "v2giant" in args.arch:
        num_heads = 24
        num_block = 40

    if "reg" in args.arch:
        num_register = 4

    if "14" in args.arch:
        patch_size = 14
    elif "16" in args.arch:
        patch_size = 16

    model, autocast_dtype = setup_and_build_model(args)
    for p in model.parameters():
        p.requires_grad = False
        
    # ============ preparing data ... ============
    transform_data_attnmap = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size, interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    result_dir = os.path.join(args.output_dir, "attention_average", f"block{args.evalblock_idx_from_end}fromend", args.model_type)
    if args.attn_row is not None and args.attn_column is not None:
        result_dir = os.path.join(result_dir, f"row{args.attn_row}column{args.attn_column}")
    os.makedirs(result_dir, exist_ok=True)


    dataset_val = make_dataset(
        dataset_str=f"ImageNet:split=VAL:root={args.root_path}:extra={args.extra_path}",
        transform=transform_data_attnmap,
    )

    data_loader_val = make_data_loader(
        dataset=dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    print(f"Data loaded with {len(dataset_val)} val imgs.")


    featmap_pos0 = args.image_size[0] // patch_size
    featmap_pos1 = args.image_size[1] // patch_size

    attn_mean = v2_extract_attention_mean_select(model=model, args=args, data_loader=data_loader_val)

    attn_mean = attn_mean.reshape(num_heads, featmap_pos0, featmap_pos1)
    attn_mean_np = nn.functional.interpolate(attn_mean.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    for j in range(num_heads):
        if args.attn_row is None or args.attn_column is None:
            fname = os.path.join(result_dir, "attn-querymean" + str(j) + "_size" + str(args.image_size[0]) + "_cls" + ".png")
        else:    
            fname = os.path.join(result_dir, "attn-querymean" + str(j) + "_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png")
        plt.imsave(fname=fname, arr=attn_mean_np[j], format='png')
        print(f"{fname} saved.")

    attn_mean_headmean_np = attn_mean_np.mean(axis=0)
    if args.attn_row is None or args.attn_column is None:
        fname = os.path.join(result_dir, "attn-headquerymean_size" + str(args.image_size[0]) + "_cls" + ".png")
    else:    
        fname = os.path.join(result_dir, "attn-headquerymean_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png")
    plt.imsave(fname=fname, arr=attn_mean_headmean_np, format='png')
    print(f"{fname} saved.")

    row=2
    col=num_heads//row
    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(6,2))
    for j in range(num_heads):
        ax[j//col,j%col].set_title(j+1, fontsize=16)
        ax[j//col,j%col].axes.xaxis.set_visible(False)
        ax[j//col,j%col].axes.yaxis.set_visible(False)
        ax[j//col,j%col].imshow(attn_mean_np[j])
    plt.tight_layout()
    if args.attn_row is None or args.attn_column is None:
        plt.savefig(os.path.join(result_dir, f"attn-querymeanlist_{row}row" + "_size" + str(args.image_size[0]) + "_cls" + ".png"), transparent=True)
    else:
        plt.savefig(os.path.join(result_dir, f"attn-querymeanlist_{row}row" + "_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png"), transparent=True)
    plt.close()

    print("Done!")
    endtime = time.time()
    print(f"time: {endtime - starttime}")        