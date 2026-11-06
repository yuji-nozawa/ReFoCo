# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified by Kioxia Corporation, 2026.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils

import os
import logging
logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, model_type):
    assert model_type == "teacher" or model_type == "student"
    if model_type == "teacher":
        model, _ = build_model_from_cfg(config, only_teacher=True)
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    else:
        model, _, _ = build_model_from_cfg(config, only_teacher=False) #student model
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "student")

    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)

    model = build_model_for_eval(config, args.pretrained_weights, args.model_type)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
