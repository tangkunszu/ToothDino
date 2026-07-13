# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random

import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
        bias_mode="uniform",
        bias_strength=2.5,
        focus_center_x=0.5,
        focus_center_y=0.5,
        focus_spread_x=0.35,
        focus_spread_y=0.2,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        self.bias_mode = bias_mode
        self.bias_strength = bias_strength
        self.focus_center_x = focus_center_x
        self.focus_center_y = focus_center_y
        self.focus_spread_x = focus_spread_x
        self.focus_spread_y = focus_spread_y
        self.bias_map = self._build_bias_map()

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f, bias=%s)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
            self.bias_mode,
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _build_bias_map(self):
        if self.bias_mode == "uniform":
            return np.ones((self.height, self.width), dtype=np.float32)

        ys = np.linspace(0.0, 1.0, self.height, endpoint=False) + 0.5 / self.height
        xs = np.linspace(0.0, 1.0, self.width, endpoint=False) + 0.5 / self.width
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        gaussian_x = np.exp(-0.5 * ((xx - self.focus_center_x) / max(self.focus_spread_x, 1e-6)) ** 2)
        gaussian_y = np.exp(-0.5 * ((yy - self.focus_center_y) / max(self.focus_spread_y, 1e-6)) ** 2)
        bias = 1.0 + self.bias_strength * gaussian_x * gaussian_y
        return bias.astype(np.float32)

    def _sample_top_left(self, h, w, bias_map=None):
        max_y = self.height - h
        max_x = self.width - w
        active_bias_map = self.bias_map if bias_map is None else bias_map
        if self.bias_mode == "uniform" and bias_map is None or max_y <= 0 or max_x <= 0:
            return random.randint(0, max_y), random.randint(0, max_x)

        scores = []
        coords = []
        for top in range(max_y + 1):
            for left in range(max_x + 1):
                coords.append((top, left))
                scores.append(active_bias_map[top : top + h, left : left + w].mean())
        scores = np.asarray(scores, dtype=np.float64)
        scores /= scores.sum()
        index = np.random.choice(len(coords), p=scores)
        return coords[index]

    def _mask(self, mask, max_mask_patches, bias_map=None):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top, left = self._sample_top_left(h, w, bias_map=bias_map)
                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0, bias_map=None):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches, bias_map=bias_map)
            if delta == 0:
                break
            mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches, bias_map=bias_map)

    def complete_mask_randomly(self, mask, num_masking_patches, bias_map=None):
        m2 = mask.flatten()
        remaining = num_masking_patches - m2.sum()
        if remaining <= 0:
            return m2.reshape(mask.shape)

        valid_indices = np.where(~m2)[0]
        if self.bias_mode == "uniform" and bias_map is None:
            to_add = np.random.choice(valid_indices, size=remaining, replace=False)
        else:
            active_bias_map = self.bias_map if bias_map is None else bias_map
            weights = active_bias_map.flatten()[valid_indices].astype(np.float64)
            weights /= weights.sum()
            to_add = np.random.choice(valid_indices, size=remaining, replace=False, p=weights)
        m2[to_add] = True
        return m2.reshape(mask.shape)
