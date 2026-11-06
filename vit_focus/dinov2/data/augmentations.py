# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions of this file are derived from torchvision.
# torchvision is licensed under the BSD 3-Clause License.
# Copyright (c) Soumith Chintala 2016.
# 
# This file contains modifications by Kioxia Corporation, 2026.
#
# The DINOv2-derived portions are licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# The torchvision-derived portions are subject to the BSD 3-Clause License.
# See LICENCES/BSD-3-Clause-torchvision.txt.


import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Optional, Union



import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

try:
    import accimage
except ImportError:
    accimage = None

import logging

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

# Adapted from torchvision.transforms.transforms.RandomResizedCrop.
# Modified by Kioxia Corporation, 2026.
class RandomResizedCropWithInfo(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: list[float], ratio: list[float]) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), i, j, h, w

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        patch_size,
        global_crops_size=224,
        local_crops_size=96,
        do_phs_inside=False,
        do_phs_outside=False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.patch_len = global_crops_size // patch_size
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.do_phs_inside = do_phs_inside
        self.do_phs_outside = do_phs_outside

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"patch_size: {patch_size}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"do_phs_inside: {do_phs_inside}")
        logger.info(f"do_phs_outside: {do_phs_outside}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # random resized crop for phs
        self.phs_augmentation_local_withinfo = RandomResizedCropWithInfo(
            local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.phs_resize = transforms.Resize(
            (global_crops_size, global_crops_size), interpolation=transforms.InterpolationMode.BICUBIC
        )
                    
        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        self.totensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.normalize_fortensor = transforms.Compose(
            [
                make_normalize_transform(),
            ]
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        if self.do_phs_inside or self.do_phs_outside:
            im3_base, crop_i, crop_j, crop_h, crop_w = self.phs_augmentation_local_withinfo(image)

        if self.do_phs_inside:
            # phs local crops for teacher
            output["phs_local_crops"] = [self.normalize(im3_base)]

        if self.do_phs_outside:
            # phs masked images for teacher
            im4_base = self.totensor(image)
            im4_base[..., crop_i:crop_i + crop_h, crop_j:crop_j + crop_w] = 0
            im4_base = self.phs_resize(im4_base)
            output["phs_masked_images"] = [self.normalize_fortensor(im4_base)]

        if self.do_phs_inside or self.do_phs_outside:
            # phs resized images for student and visual prompts
            im5_base = self.phs_resize(image)
            output["phs_images_student"] = [self.normalize(im5_base)]

            img_h = image.height
            img_w = image.width

            crop_y1 = crop_i
            crop_y2 = crop_i + crop_h - 1
            crop_x1 = crop_j
            crop_x2 = crop_j + crop_w - 1

            patchid_y1 = math.floor(self.patch_len * crop_y1 / img_h)
            patchid_y2 = math.floor(self.patch_len * crop_y2 / img_h)
            patchid_x1 = math.floor(self.patch_len * crop_x1 / img_w)
            patchid_x2 = math.floor(self.patch_len * crop_x2 / img_w)

            output["phs_box"] = [torch.tensor((patchid_y1, patchid_y2, patchid_x1, patchid_x2), dtype=torch.int64)]

        return output
