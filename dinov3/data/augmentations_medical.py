# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger("dinov3")


class MedicalImageAugmentation:
    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: Tuple[float, float] = (0, 5),
        horizontal_flip: float = 0.5,
        enabled: bool = True,
    ):
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.enabled = enabled

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img
        img = self._gamma_correction(img)
        img = self._contrast_enhancement(img)
        img = self._brightness_adjustment(img)
        if random.random() < self.horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self._small_rotation(img)
        return img

    def _gamma_correction(self, img: Image.Image) -> Image.Image:
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = np.clip(img_np, 0, 1)
        return Image.fromarray((img_np * 255).astype(np.uint8))

    def _contrast_enhancement(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img

    def _brightness_adjustment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            img = ImageEnhance.Brightness(img).enhance(factor)
        return img

    def _small_rotation(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
        return img
