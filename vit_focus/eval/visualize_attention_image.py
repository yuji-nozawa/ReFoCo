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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as pth_transforms

from dinov2.utils import bool_flag as utilsbool
from dinov2.logging import helpers
from dinov2.eval.setup import setup_and_build_model

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

    return attn

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize attention maps")
    parser.add_argument("--arch", choices=["v2smallreg16", "v2basereg16", "v2largereg16", "v2giantreg16"], default="v2large16", type=str, help="Architecture")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
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


    # # ============ building network ... ============
    model, autocast_dtype = setup_and_build_model(args)

    for p in model.parameters():
        p.requires_grad = False

    if args.attn_row is not None and args.attn_column is not None:
        attn_component = args.attn_row * args.image_size[0]//patch_size + args.attn_column + 1 + num_register #+1 for [cls] token

    #output directory for each image

    image_dir = os.path.join(args.output_dir, "attention_image", f"block{args.evalblock_idx_from_end}fromend", args.model_type, os.path.splitext(os.path.basename(args.image_path))[0])
    if args.attn_row is not None and args.attn_column is not None:
        image_dir = os.path.join(image_dir, f"row{args.attn_row}column{args.attn_column}")

    # open image
    if os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    transform1 = pth_transforms.Compose([
        pth_transforms.Resize((224,224), interpolation=3),
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
    ])

    transform2 = pth_transforms.Compose([
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img0 = transform1(img)
    img = transform1(img)
    img = transform2(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size


    attentions = get_attentions_select(model, img, num_block, phsblock_idx_from_end, args)

    nh = attentions.shape[1] # number of heads

    if args.attn_row is None or args.attn_column is None:
        attentions = attentions[0, :, 0, num_register + 1:].reshape(nh, -1)
    else:
        attentions = attentions[0, :, attn_component, num_register + 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(image_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img0, normalize=True, scale_each=True), os.path.join(image_dir, "img"  + "_size" + str(args.image_size[0]) + ".png"))
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(image_dir, "img_transformed"  + "_size" + str(args.image_size[0]) + ".png"))
    for j in range(nh):
        if args.attn_row is None or args.attn_column is None:
            fname = os.path.join(image_dir, "attn-head" + str(j) + "_size" + str(args.image_size[0]) + "_cls" + ".png")
        else:    
            fname = os.path.join(image_dir, "attn-head" + str(j) + "_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    mean_attentions = attentions.mean(axis=0)
    if args.attn_row is None or args.attn_column is None:
        fname = os.path.join(image_dir, "attnmean_size" + str(args.image_size[0]) + "_cls" + ".png")
    else:    
        fname = os.path.join(image_dir, "attnmean_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png")
    plt.imsave(fname=fname, arr=mean_attentions, format='png')
    print(f"{fname} saved.")

    mean_attentions_normarized = (mean_attentions - mean_attentions.min()) / (mean_attentions.max() - mean_attentions.min())
    threshold_mean_attentions = mean_attentions_normarized.mean()
    mask_mean_attentions = mean_attentions_normarized > threshold_mean_attentions

    if args.attn_row is None or args.attn_column is None:
        fname = os.path.join(image_dir, "mask_attnmean_size" + str(args.image_size[0]) + "_cls" + ".png")
    else:    
        fname = os.path.join(image_dir, "mask_attnmean_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png")
    plt.imsave(fname=fname, arr=mask_mean_attentions, format='png')
    print(f"{fname} saved.")

    row=2
    col=nh//row
    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(6,2))
    for j in range(nh):
        ax[j//col,j%col].set_title(j+1, fontsize=16)
        ax[j//col,j%col].axes.xaxis.set_visible(False)
        ax[j//col,j%col].axes.yaxis.set_visible(False)
        ax[j//col,j%col].imshow(attentions[j])
    plt.tight_layout()
    if args.attn_row is None or args.attn_column is None:
        plt.savefig(os.path.join(image_dir, f"attnlist_{row}row" + "_size" + str(args.image_size[0]) + "_cls" + ".png"), transparent=True)
    else:
        plt.savefig(os.path.join(image_dir, f"attnlist_{row}row" + "_size" + str(args.image_size[0]) + "_row" + str(args.attn_row) + "_column" + str(args.attn_column) + ".png"), transparent=True)
    plt.close()