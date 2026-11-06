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
from dinov2.data import SamplerType, make_data_loader, make_dataset, CustomDataset

from omegaconf import OmegaConf 
import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def extract_features(args, model, data_loader, hs_flag=False, box=None, is_outside=False):
    metric_logger = helpers.MetricLogger(delimiter="  ")

    features = None

    totalcount = 0
    indexnumbercount = 0

    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        indexnumber = torch.zeros([index.size(0)],dtype=torch.int64)
        indexnumber = indexnumber.cuda(non_blocking=True)
        for indexnumbercount in range(index.size(0)):
            indexnumber[indexnumbercount] = totalcount + indexnumbercount

        feats = model(samples, masks=None, is_training=False, hs_flag=hs_flag, box=box, is_outside=is_outside)

        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        totalcount += index.size(0)

        features.index_copy_(0, indexnumber, feats)

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract PHS features")
    parser.add_argument("--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument("--arch", choices=["v2smallreg16", "v2basereg16", "v2largereg16", "v2giantreg16"], default="v2large16", type=str, help="Architecture")
    parser.add_argument("--dataset_type", choices=["ImageNet", "Custom"], default="ImageNet", type=str, help="Dataset type (default: ImageNet)")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--extra_path", type=str)
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--model_type", choices=["teacher", "student"], default="teacher", type=str, help="Type of model to build in setup_and_build_model. Set to 'student' when using a student model.")
    parser.add_argument("--hs_flag", default=False, type=utilsbool.bool_flag, 
        help="If True, PHS is applied during feature extraction."
    )
    parser.add_argument("--is_outside", default=False, type=utilsbool.bool_flag, 
        help="If False, the inside of the box is used as the visual prompt; if True, the outside is used."
    )
    parser.add_argument("--save_features", action="store_true", help="Save extracted features.")
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

    if args.dataset_type == "ImageNet":
        if args.extra_path is None:
            parser.error("--extra_path is required for ImageNet")
        dataset_val = make_dataset(
            dataset_str=f"ImageNet:split=VAL:root={args.root_path}:extra={args.extra_path}",
            transform=transform_data_attnmap,
        )
    elif args.dataset_type == "Custom":
        dataset_val = CustomDataset(
            root_path=args.root_path,
            transform=transform_data_attnmap,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

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

    box = None
    if args.hs_flag:
        # This script currently uses a placeholder box example.
        patchid_y1, patchid_y2, patchid_x1, patchid_x2 = 2, 3, 5, 7
        box_example = torch.tensor((patchid_y1, patchid_y2, patchid_x1, patchid_x2), dtype=torch.int64)
        box = torch.stack([box_example for _ in range(len(dataset_val))])
        box = box.cuda(non_blocking=True)

    val_features = extract_features(args, model, data_loader_val, hs_flag=args.hs_flag, box=box, is_outside=args.is_outside)

    print(f"{val_features.shape=}")


    if args.save_features:
        result_dir = os.path.join(args.output_dir, "features", args.model_type)
        os.makedirs(result_dir, exist_ok=True)

        if args.hs_flag:
            region = "outside" if args.is_outside else "inside"
            filename = f"features_hs_{region}.pth"
        else:
            filename = "features.pth"

        torch.save(val_features.cpu(), os.path.join(result_dir, filename))
        print("Features saved.")


    print("Done!")
    endtime = time.time()
    print(f"time: {endtime - starttime}")        