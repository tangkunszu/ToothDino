# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import json
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

try:
    from dinov3.data.augmentations_medical import MedicalImageAugmentation
except ImportError:
    MedicalImageAugmentation = None

try:
    from dinov3.data.augmentations_spectral import FrequencySpectrumAugmentation
except ImportError:
    FrequencySpectrumAugmentation = None

logger = logging.getLogger("dinov3")


# Smallest crop side (in source pixels) n-TCC will produce. Kept well below
# local_crops_size so the requested crop geometry is never silently inflated;
# sub-resolution crops are simply upsampled to local_crops_size, as in DINOv3.
_LOCAL_CROP_MIN_PIXELS = 16


class AdditiveGaussianNoise(nn.Module):
    def __init__(self, p=0.0, std_range=(0.005, 0.02)):
        super().__init__()
        self.p = p
        self.std_range = std_range

    def forward(self, image):
        if self.p <= 0 or random.random() >= self.p:
            return image
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(image) * std
        return (image + noise).clamp(0.0, 1.0)


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        global_crops_ratio=None,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        output_channels=3,
        medical_augmentation=False,
        gamma_range=(0.8, 1.2),
        contrast_range=(0.85, 1.15),
        brightness_range=(0.9, 1.1),
        rotation_range=(0, 5),
        augmentation_mode="natural",
        local_crop_strategy="random",
        local_quadrant_ratio=0.5,
        local_arch_focus_strength=0.7,
        local_crop_jitter=0.15,
        local_crop_min_scale=0.6,
        local_crop_max_scale=1.0,
        band_aware_fast_crop_min_ratio=0.22,
        band_aware_fast_crop_max_ratio=0.31,
        band_aware_fast_upper_center_offset=0.09,
        band_aware_fast_lower_center_offset=0.09,
        band_aware_fast_use_official_area_scale=False,
        representative_tooth_aware_attempts=80,
        representative_tooth_cache_path=None,
        direct_center_jitter=0.03,
        direct_random_scale=True,
        xray_noise_probability=0.15,
        xray_noise_std=(0.005, 0.02),
        spectral_augmentation_enabled=False,
        spectral_augmentation_p=0.5,
        spectral_mask_low_range=(0.08, 0.40),
        spectral_mask_width_range=(0.10, 0.30),
        spectral_attenuation_range=(0.30, 0.85),
        anatomy_guided_masking_enabled=False,
        anatomy_guided_masking_source="image_intensity",
        anatomy_guided_masking_mix_with_static_prior=0.5,
        anatomy_guided_masking_epsilon=1e-6,
        layered_masking_enabled=False,
        layered_mask_core_weight=0.55,
        layered_mask_context_weight=0.30,
        layered_mask_background_weight=0.15,
        anatomy_guided_masking_full_prior_max_side=512,
        anatomy_guided_masking_projection=None,
        anatomy_guided_masking_tooth_prior=None,
        global_crop_strategy="random",
        global_tooth_center_jitter=0.035,
        global_tooth_center_scale=(0.56, 0.60),
        hierarchical_transition_ratio=0.0,
        view_aware_augmentation=False,
        global_jitter_scale=0.75,
        local_jitter_scale=1.15,
        global_noise_scale=0.75,
        local_noise_scale=1.25,
        blur_probability_global1=None,
        blur_probability_global2=None,
        blur_probability_local=None,
        local_policy=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_random_crop_ratio = (3.0 / 4.0, 4.0 / 3.0)
        # Aspect-ratio range of the GLOBAL crop taken from the source panorama, before it
        # is resized to the square global_crops_size. The DINOv3 default (3/4, 4/3) picks a
        # near-square region, which on a ~2:1 panorama covers only part of the arch:
        # measured over 322 real global views, the median crop spans 0.494 of the image
        # width and 0.608 of the dental arch, and only 0.3% contain >=95% of the arch.
        # DINO's local-to-global objective assumes the global view represents the whole, so
        # widening this range lets a global view actually contain the full dentition (at
        # the cost of squashing it into the square target). None keeps the DINOv3 default.
        self.global_random_crop_ratio = (
            tuple(global_crops_ratio) if global_crops_ratio else (3.0 / 4.0, 4.0 / 3.0)
        )
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.output_channels = output_channels
        self.augmentation_mode = augmentation_mode
        self.local_crop_strategy = local_crop_strategy
        self.local_quadrant_ratio = local_quadrant_ratio
        self.local_arch_focus_strength = local_arch_focus_strength
        self.local_crop_jitter = local_crop_jitter
        self.local_crop_min_scale = local_crop_min_scale
        self.local_crop_max_scale = local_crop_max_scale
        self.band_aware_fast_crop_min_ratio = band_aware_fast_crop_min_ratio
        self.band_aware_fast_crop_max_ratio = band_aware_fast_crop_max_ratio
        self.band_aware_fast_upper_center_offset = band_aware_fast_upper_center_offset
        self.band_aware_fast_lower_center_offset = band_aware_fast_lower_center_offset
        self.band_aware_fast_use_official_area_scale = band_aware_fast_use_official_area_scale
        self.representative_tooth_aware_attempts = int(max(representative_tooth_aware_attempts, 1))
        self.direct_center_jitter = direct_center_jitter
        self.direct_random_scale = direct_random_scale
        self.representative_tooth_cache_path = representative_tooth_cache_path
        self.representative_tooth_cache = self._load_representative_tooth_cache(representative_tooth_cache_path)
        self.accepts_image_path = True
        self.anatomy_guided_masking_enabled = anatomy_guided_masking_enabled
        self.anatomy_guided_masking_source = anatomy_guided_masking_source
        self.anatomy_guided_masking_mix_with_static_prior = anatomy_guided_masking_mix_with_static_prior
        self.anatomy_guided_masking_epsilon = anatomy_guided_masking_epsilon
        self.layered_masking_enabled = layered_masking_enabled
        self.layered_mask_core_weight = layered_mask_core_weight
        self.layered_mask_context_weight = layered_mask_context_weight
        self.layered_mask_background_weight = layered_mask_background_weight
        self.anatomy_guided_masking_full_prior_max_side = int(max(anatomy_guided_masking_full_prior_max_side, 64))
        self.anatomy_guided_masking_projection = anatomy_guided_masking_projection or {}
        self.anatomy_guided_masking_tooth_prior = anatomy_guided_masking_tooth_prior or {}
        self._abmv2_prior_cache = {}
        self.global_crop_strategy = global_crop_strategy
        self.global_tooth_center_jitter = float(global_tooth_center_jitter)
        self.global_tooth_center_scale = tuple(global_tooth_center_scale)
        self.hierarchical_transition_ratio = hierarchical_transition_ratio
        self.view_aware_augmentation = view_aware_augmentation
        self.global_jitter_scale = global_jitter_scale
        self.local_jitter_scale = local_jitter_scale
        self.global_noise_scale = global_noise_scale
        self.local_noise_scale = local_noise_scale
        local_policy = local_policy or {}
        self.local_policy = {
            "random_foreground_number": int(local_policy.get("random_foreground_number", 4)),
            "fine_dental_number": int(local_policy.get("fine_dental_number", 4)),
            "anchor_dental_number": int(local_policy.get("anchor_dental_number", 4)),
            "random_foreground_scale": tuple(local_policy.get("random_foreground_scale", (0.05, 0.32))),
            "fine_dental_scale": tuple(local_policy.get("fine_dental_scale", (0.05, 0.12))),
            "anchor_dental_scale": tuple(local_policy.get("anchor_dental_scale", (0.12, 0.28))),
            "min_foreground_ratio": float(local_policy.get("min_foreground_ratio", 0.30)),
            "center_jitter": float(local_policy.get("center_jitter", 0.15)),
            "avoid_black_background": bool(local_policy.get("avoid_black_background", True)),
            "max_resample_attempts": int(local_policy.get("max_resample_attempts", 20)),
        }
        self.foreground_intensity_threshold = 0.05
        self.anchor_regions = (
            (0.18, 0.55),
            (0.82, 0.55),
            (0.40, 0.32),
            (0.60, 0.68),
            (0.08, 0.50),
            (0.92, 0.50),
            (0.50, 0.20),
            (0.50, 0.50),
        )

        # --- Frequency Spectrum Augmentation (tensor-domain, applied in _finalize_tensor) ---
        self.spectral_augmentation = None
        if spectral_augmentation_enabled and FrequencySpectrumAugmentation is not None:
            self.spectral_augmentation = FrequencySpectrumAugmentation(
                p=spectral_augmentation_p,
                mask_low_range=spectral_mask_low_range,
                mask_width_range=spectral_mask_width_range,
                attenuation_range=spectral_attenuation_range,
                per_channel=False,
            )
            logger.info(
                "FrequencySpectrumAugmentation enabled: p=%.2f, low_range=%s, width_range=%s, atten=%s",
                spectral_augmentation_p,
                spectral_mask_low_range,
                spectral_mask_width_range,
                spectral_attenuation_range,
            )
        elif spectral_augmentation_enabled:
            logger.warning("spectral_augmentation_enabled=True but FrequencySpectrumAugmentation is unavailable.")

        self.medical_augmentation = None
        if medical_augmentation and MedicalImageAugmentation is not None:
            self.medical_augmentation = MedicalImageAugmentation(
                gamma_range=gamma_range,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
                rotation_range=rotation_range,
                horizontal_flip=0.0,
                enabled=True,
            )
        elif medical_augmentation:
            logger.warning("Medical augmentation enabled but the medical augmentation module is unavailable.")

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"global_crops_ratio: {self.global_random_crop_ratio}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"augmentation_mode: {augmentation_mode}")
        logger.info(f"local_crop_strategy: {local_crop_strategy}")
        if local_crop_strategy in {
            "dental_hierarchical_multiscale_fast",
            "dental_hierarchical_original_compact",
            "dental_hierarchical_original_compact_plus2",
            "dental_hybrid_random4_band4",
        }:
            logger.info(
                "band_aware_fast_crop_ratio: (%s, %s)",
                self.band_aware_fast_crop_min_ratio,
                self.band_aware_fast_crop_max_ratio,
            )
            logger.info(
                "band_aware_fast_center_offset: (upper=%s, lower=%s)",
                self.band_aware_fast_upper_center_offset,
                self.band_aware_fast_lower_center_offset,
            )
            logger.info(
                "band_aware_fast_use_official_area_scale: %s",
                self.band_aware_fast_use_official_area_scale,
            )
        if local_crop_strategy == "dental_representative_tooth_aware":
            logger.info("representative_tooth_aware_attempts: %s", self.representative_tooth_aware_attempts)
        if self.representative_tooth_cache:
            logger.info(
                "representative_tooth_cache: %s entries from %s",
                len(self.representative_tooth_cache),
                representative_tooth_cache_path,
            )
        if local_crop_strategy == "foreground_dental_mixed":
            logger.info(f"local_policy: {self.local_policy}")
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)
        self.horizontal_flip_p = 0.5 if horizontal_flips else 0.0
        self.global_random_crop = v2.RandomResizedCrop(
            global_crop_max_size,
            scale=global_crops_scale,
            interpolation=v2.InterpolationMode.BICUBIC,
        )

        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = v2.Resize(global_crops_size, interpolation=v2.InterpolationMode.BICUBIC)
            else:
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    ratio=self.local_random_crop_ratio,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=self.horizontal_flip_p),
            ]
        )

        if augmentation_mode == "dental_xray":
            global_jitter_strength = global_jitter_scale if view_aware_augmentation else 1.0
            local_jitter_strength = local_jitter_scale if view_aware_augmentation else 1.0
            global_noise_probability = xray_noise_probability * (global_noise_scale if view_aware_augmentation else 1.0)
            local_noise_probability = xray_noise_probability * (local_noise_scale if view_aware_augmentation else 1.0)

            self.global_color_jittering = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.15 * global_jitter_strength,
                                contrast=0.2 * global_jitter_strength,
                            )
                        ],
                        p=min(0.5, 0.3 * global_jitter_strength),
                    ),
                ]
            )
            self.local_color_jittering = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.15 * local_jitter_strength,
                                contrast=0.2 * local_jitter_strength,
                            )
                        ],
                        p=min(0.65, 0.3 * local_jitter_strength),
                    ),
                ]
            )
            # Blur probabilities are configurable because the shipped defaults contradict
            # the weak-global / strong-local intent: global view 1 blurs at 0.45 while the
            # local branch blurs at 0.40. That 0.45/0.15 split is inherited from DINOv3,
            # where the asymmetry exists to make the two teacher views differ, not to make
            # the global branch weak. Leaving these at None keeps the historical values.
            blur_p_global1 = 0.45 if view_aware_augmentation else 0.6
            blur_p_global2 = 0.15 if view_aware_augmentation else 0.2
            blur_p_local = 0.4 if view_aware_augmentation else 0.3
            if blur_probability_global1 is not None:
                blur_p_global1 = float(blur_probability_global1)
            if blur_probability_global2 is not None:
                blur_p_global2 = float(blur_probability_global2)
            if blur_probability_local is not None:
                blur_p_local = float(blur_probability_local)
            logger.info(
                "blur probability: global1=%.2f global2=%.2f local=%.2f",
                blur_p_global1,
                blur_p_global2,
                blur_p_local,
            )
            global_transfo1_extra = v2.Compose([GaussianBlur(p=blur_p_global1)])
            global_transfo2_extra = v2.Compose([GaussianBlur(p=blur_p_global2)])
            local_transfo_extra = v2.Compose([GaussianBlur(p=blur_p_local)])
        else:
            color_jittering = v2.Compose(
                [
                    v2.RandomApply(
                        [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8,
                    ),
                    v2.RandomGrayscale(p=0.2),
                ]
            )
            global_transfo1_extra = GaussianBlur(p=1.0)
            global_transfo2_extra = v2.Compose([GaussianBlur(p=0.1), v2.RandomSolarize(threshold=128, p=0.2)])
            local_transfo_extra = GaussianBlur(p=0.5)
            self.global_color_jittering = color_jittering
            self.local_color_jittering = color_jittering
            global_noise_probability = 0.0
            local_noise_probability = 0.0

        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.global_tensor_noise = AdditiveGaussianNoise(p=global_noise_probability, std_range=xray_noise_std)
        self.local_tensor_noise = AdditiveGaussianNoise(p=local_noise_probability, std_range=xray_noise_std)
        self.normalize = make_normalize_transform(mean=mean, std=std)

        if self.share_color_jitter:
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra])
            self.local_transfo = v2.Compose([local_transfo_extra])
        else:
            self.global_transfo1 = v2.Compose([resize_global, self.global_color_jittering, global_transfo1_extra])
            self.global_transfo2 = v2.Compose([resize_global, self.global_color_jittering, global_transfo2_extra])
            self.local_transfo = v2.Compose([self.local_color_jittering, local_transfo_extra])

    def __call__(self, image, image_path=None):
        if self.medical_augmentation is not None:
            image = self.medical_augmentation(image)

        output = {
            "weak_flag": True,
            "global_flip_labels": [],
            "local_side_labels": [],
            "local_region_labels": [],
            "local_crop_boxes": [],
            "local_source_global_indices": [],
        }

        if self.share_color_jitter:
            image = self.global_color_jittering(image)

        im1_base, im1_flipped, im1_box = self._sample_global_crop_with_meta(image, view_index=0)
        global_crop_1_transf = self._finalize_tensor(self.global_transfo1(im1_base), is_local=False)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base, im2_flipped, im2_box = self._sample_global_crop_with_meta(image, view_index=1)
        global_crop_2_transf = self._finalize_tensor(self.global_transfo2(im2_base), is_local=False)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_flip_labels"] = [float(im1_flipped), float(im2_flipped)]
        if self.anatomy_guided_masking_enabled:
            output["mask_bias_maps"] = [
                self._build_mask_bias_map(
                    im1_base,
                    global_crop_1.shape[-2] // self.patch_size,
                    original_image=image,
                    crop_box=im1_box,
                    flipped=im1_flipped,
                    image_path=image_path,
                ),
                self._build_mask_bias_map(
                    im2_base,
                    global_crop_2.shape[-2] // self.patch_size,
                    original_image=image,
                    crop_box=im2_box,
                    flipped=im2_flipped,
                    image_path=image_path,
                ),
            ]

        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self._finalize_tensor(im1_base, is_local=False),
                self._finalize_tensor(im2_base, is_local=False),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self._finalize_tensor(self.resize_gram_teacher(im1_base), is_local=False)
                gram_crop_2 = self._finalize_tensor(self.resize_gram_teacher(im2_base), is_local=False)
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        if self.local_crops_subset_of_global_crops:
            _local_crops = [
                self._finalize_tensor(self.local_transfo(im1_base), is_local=True)
                for _ in range(self.local_crops_number // 2)
            ] + [
                self._finalize_tensor(self.local_transfo(im2_base), is_local=True)
                for _ in range(self.local_crops_number // 2)
            ]
            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))
            output["local_crops"] = local_crops
            output["offsets"] = offsets
            output["local_side_labels"] = [1 for _ in local_crops]
            output["local_region_labels"] = [0 for _ in local_crops]
            output["local_crop_boxes"] = [(float(rx), float(ry), float(rx + ls), float(ry + ls)) for rx, ry in offsets]
            output["local_source_global_indices"] = [
                0 if i < (self.local_crops_number // 2) else 1 for i in range(len(local_crops))
            ]
        else:
            if self.local_crop_strategy == "dental_hierarchical":
                local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices = self._sample_hierarchical_local_crops(
                    [im1_base, im2_base]
                )
            elif self.local_crop_strategy == "dental_hierarchical_from_original":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hierarchical_local_crops([image])
            elif self.local_crop_strategy == "dental_hierarchical_multilevel":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hierarchical_multilevel_local_crops([im1_base, im2_base])
            elif self.local_crop_strategy == "dental_hierarchical_2plus4":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hierarchical_2plus4_local_crops([im1_base, im2_base])
            elif self.local_crop_strategy == "dental_hierarchical_12layer":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hierarchical_12layer_local_crops([im1_base, im2_base])
            elif self.local_crop_strategy == "dental_hierarchical_10layer":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hierarchical_10layer_local_crops(image)
            elif self.local_crop_strategy == "dental_hierarchical_original_compact":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_original_compact_hierarchical_local_crops(image, self.local_crops_number)
            elif self.local_crop_strategy == "dental_hierarchical_original_compact_plus2":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_original_compact_hierarchical_plus2_local_crops(image)
            elif self.local_crop_strategy == "dental_hierarchical_multiscale":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_original_compact_hierarchical_multiscale_local_crops(image)
            elif self.local_crop_strategy == "dental_hierarchical_multiscale_fast":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_original_compact_hierarchical_multiscale_fast_local_crops(image)
            elif self.local_crop_strategy == "dental_representative_tooth_aware":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_representative_tooth_aware_local_crops(image)
            elif self.local_crop_strategy == "tcc_legacy":
                # Pre-n-TCC geometry, frozen verbatim for the TCC vs n-TCC ablation.
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_tcc_legacy_local_crops(image, image_path=image_path)
            elif self.local_crop_strategy in (
                # n-TCC: tooth-centric local crops placed on representative tooth-row
                # centers, with crop side = sqrt(area_scale) * H -- anchored to image
                # height rather than image area so the ~2:1 panoramic aspect ratio does not
                # inflate the crop, and taken straight from the source panorama. Area,
                # aspect and center jitter are redrawn per crop, so the local branch never
                # sees pixel-identical views across epochs. Set direct_center_jitter=0.0
                # and direct_random_scale=false for the deterministic ablation.
                "n_tcc",
                "dental_representative_tooth_direct",  # legacy alias
                "dental_stochastic_tooth_centric",  # legacy alias
                "dental_representative_tooth_direct_jitter",  # legacy alias
            ):
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_representative_tooth_direct_local_crops(
                    image,
                    image_path=image_path,
                    center_jitter=self.direct_center_jitter,
                    random_scale=self.direct_random_scale,
                )
            elif self.local_crop_strategy == "dental_representative_tooth_adaptive_direct":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_representative_tooth_adaptive_direct_local_crops(image)
            elif self.local_crop_strategy == "dental_representative_tooth_static_direct":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_representative_tooth_static_direct_local_crops(image)
            elif self.local_crop_strategy == "dental_hybrid_random4_band4":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_hybrid_random_band_local_crops(image)
            elif self.local_crop_strategy == "dental_hierarchical_compact":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_compact_hierarchical_local_crops([im1_base, im2_base])
            elif self.local_crop_strategy == "dental_hierarchical_balanced":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_balanced_hierarchical_local_crops([im1_base, im2_base])
            elif self.local_crop_strategy == "anatomy_guided":
                local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices = self._sample_anatomy_guided_local_crops(
                    [im1_base, im2_base]
                )
            elif self.local_crop_strategy == "foreground_dental_mixed":
                (
                    local_crops,
                    local_side_labels,
                    local_region_labels,
                    local_crop_boxes,
                    local_source_global_indices,
                ) = self._sample_foreground_dental_mixed_local_crops([im1_base, im2_base])
            else:
                local_crops = [
                    self._finalize_tensor(self.local_transfo(self.geometric_augmentation_local(image)), is_local=True)
                    for _ in range(self.local_crops_number)
                ]
                local_side_labels = [1 for _ in local_crops]
                local_region_labels = [0 for _ in local_crops]
                local_crop_boxes = [
                    (0.0, 0.0, float(self.global_crops_size), float(self.global_crops_size)) for _ in local_crops
                ]
                local_source_global_indices = [i % 2 for i in range(len(local_crops))]
            output["local_crops"] = local_crops
            output["local_side_labels"] = local_side_labels
            output["local_region_labels"] = local_region_labels
            output["local_crop_boxes"] = local_crop_boxes
            output["local_source_global_indices"] = local_source_global_indices
            output["offsets"] = ()

        return output

    def _load_representative_tooth_cache(self, cache_path):
        if not cache_path:
            return {}
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", data) if isinstance(data, dict) else {}
        cache = {}
        for key, value in entries.items():
            normalized_key = str(key).replace("\\", "/")
            centers = value.get("centers", value) if isinstance(value, dict) else value
            if not centers:
                continue
            cache[normalized_key] = [tuple(float(v) for v in center[:2]) for center in centers]
        return cache

    def _get_cached_representative_tooth_centers(self, image_path):
        if not image_path or not self.representative_tooth_cache:
            return None
        key = str(image_path).replace("\\", "/")
        centers = self.representative_tooth_cache.get(key)
        if centers is None and key.startswith("./"):
            centers = self.representative_tooth_cache.get(key[2:])
        if centers is None:
            return None
        return centers[: self.local_crops_number]

    def _build_mask_bias_map(
        self,
        image,
        patch_grid_size,
        *,
        original_image=None,
        crop_box=None,
        flipped=False,
        image_path=None,
    ):
        if not self.anatomy_guided_masking_enabled:
            return None
        if self.anatomy_guided_masking_source == "full_image_tooth_prior_projected":
            if original_image is None or crop_box is None:
                # Fallback keeps debugging scripts that pass only a crop usable.
                original_image = image
                crop_box = (0, 0, image.size[0], image.size[1])
            return self._build_projected_tooth_prior_bias_map(
                original_image,
                crop_box,
                patch_grid_size,
                flipped=flipped,
                image_path=image_path,
            )
        if self.anatomy_guided_masking_source != "image_intensity":
            raise ValueError(
                f"Unsupported anatomy_guided_masking.source={self.anatomy_guided_masking_source}"
            )

        gray = ImageOps.grayscale(image).resize((patch_grid_size, patch_grid_size), resample=Image.BICUBIC)
        gray_np = np.asarray(gray, dtype=np.float32) / 255.0
        gray_np = gray_np - gray_np.min()
        gray_np = gray_np / max(float(gray_np.max()), self.anatomy_guided_masking_epsilon)

        ys = np.linspace(0.0, 1.0, patch_grid_size, endpoint=False) + 0.5 / patch_grid_size
        xs = np.linspace(0.0, 1.0, patch_grid_size, endpoint=False) + 0.5 / patch_grid_size
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        gaussian_x = np.exp(-0.5 * ((xx - 0.5) / 0.35) ** 2)
        gaussian_y = np.exp(-0.5 * ((yy - 0.5) / 0.2) ** 2)
        static_prior = gaussian_x * gaussian_y
        static_prior = static_prior / max(float(static_prior.max()), self.anatomy_guided_masking_epsilon)

        if self.layered_masking_enabled:
            core_prior = gray_np * static_prior
            context_prior = gray_np * np.sqrt(np.clip(static_prior, self.anatomy_guided_masking_epsilon, None))
            background_prior = np.ones_like(gray_np, dtype=np.float32)
            mixed = (
                self.layered_mask_core_weight * core_prior
                + self.layered_mask_context_weight * context_prior
                + self.layered_mask_background_weight * background_prior
            )
        else:
            mixed = (1 - self.anatomy_guided_masking_mix_with_static_prior) * gray_np + (
                self.anatomy_guided_masking_mix_with_static_prior * static_prior
            )
        mixed = mixed + self.anatomy_guided_masking_epsilon
        mixed = mixed / mixed.sum()
        return torch.from_numpy(mixed.astype(np.float32))

    def _build_projected_tooth_prior_bias_map(self, image, crop_box, patch_grid_size, *, flipped=False, image_path=None):
        prior_layers = self._get_full_image_tooth_prior_layers(image, image_path=image_path)
        core = self._project_prior_layer_to_crop_grid(prior_layers["core"], crop_box, image.size, patch_grid_size)
        context = self._project_prior_layer_to_crop_grid(prior_layers["context"], crop_box, image.size, patch_grid_size)
        background = np.ones_like(core, dtype=np.float32)

        if flipped:
            core = np.fliplr(core)
            context = np.fliplr(context)

        if self.layered_masking_enabled:
            mixed = (
                self.layered_mask_core_weight * core
                + self.layered_mask_context_weight * context
                + self.layered_mask_background_weight * background
            )
        else:
            foreground = 0.78 * core + 0.22 * context
            mixed = (
                (1 - self.anatomy_guided_masking_mix_with_static_prior) * foreground
                + self.anatomy_guided_masking_mix_with_static_prior * background
            )
        mixed = np.clip(mixed, 0, None).astype(np.float32) + self.anatomy_guided_masking_epsilon
        mixed = mixed / max(float(mixed.sum()), self.anatomy_guided_masking_epsilon)
        return torch.from_numpy(mixed.astype(np.float32))

    def _get_full_image_tooth_prior_layers(self, image, *, image_path=None):
        cache_key = None
        if image_path:
            cache_key = (str(image_path), image.size)
            cached = self._abmv2_prior_cache.get(cache_key)
            if cached is not None:
                return cached

        width, height = image.size
        max_side = self.anatomy_guided_masking_full_prior_max_side
        scale = min(1.0, float(max_side) / max(width, height, 1))
        if scale < 1.0:
            work_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            work_image = image.resize(work_size, resample=Image.BILINEAR)
        else:
            work_image = image

        gray, saliency, band = self._build_tooth_focus_maps(work_image)
        centers, _ = self._get_representative_tooth_centers_fast(work_image)
        h, w = gray.shape
        ys = np.linspace(0.0, 1.0, h, endpoint=False, dtype=np.float32) + 0.5 / h
        xs = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32) + 0.5 / w
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        tooth_cfg = self.anatomy_guided_masking_tooth_prior
        sigma_x = float(tooth_cfg.get("center_blob_sigma_x", 0.045))
        sigma_y = float(tooth_cfg.get("center_blob_sigma_y", 0.060))
        tooth_blobs = np.zeros((h, w), dtype=np.float32)
        for cx, cy in centers:
            tooth_blobs += np.exp(-0.5 * (((xx - cx) / max(sigma_x, 1e-6)) ** 2 + ((yy - cy) / max(sigma_y, 1e-6)) ** 2))

        x_left, x_right, band_y = self._stabilize_panorama_focus_band(band)
        band_center = 0.5 * (x_left + x_right)
        band_width = max(x_right - x_left, 0.10)
        arch_sigma_x_scale = float(tooth_cfg.get("arch_context_sigma_x_scale", 0.42))
        arch_sigma_y = float(tooth_cfg.get("arch_context_sigma_y", 0.16))
        arch_context = np.exp(-0.5 * ((xx - band_center) / max(arch_sigma_x_scale * band_width, 1e-6)) ** 2)
        arch_context *= np.exp(-0.5 * ((yy - band_y) / max(arch_sigma_y, 1e-6)) ** 2)

        saliency = np.clip(saliency.astype(np.float32), 0, None)
        saliency = saliency / max(float(saliency.max()), self.anatomy_guided_masking_epsilon)
        saliency_mix = float(tooth_cfg.get("saliency_mix", 0.65))
        core = tooth_blobs * ((1.0 - saliency_mix) + saliency_mix * saliency)
        context = arch_context * (0.20 + 0.80 * saliency)

        core = core + self.anatomy_guided_masking_epsilon
        context = context + self.anatomy_guided_masking_epsilon
        layers = {
            "core": (core / max(float(core.max()), self.anatomy_guided_masking_epsilon)).astype(np.float32),
            "context": (context / max(float(context.max()), self.anatomy_guided_masking_epsilon)).astype(np.float32),
            "size": work_image.size,
        }
        if cache_key is not None:
            self._abmv2_prior_cache[cache_key] = layers
        return layers

    def _project_prior_layer_to_crop_grid(self, prior, crop_box, image_size, patch_grid_size):
        image_w, image_h = image_size
        prior_h, prior_w = prior.shape
        scale_x = prior_w / max(float(image_w), 1.0)
        scale_y = prior_h / max(float(image_h), 1.0)
        x0, y0, x1, y1 = crop_box
        px0 = int(np.clip(round(float(x0) * scale_x), 0, prior_w - 1))
        py0 = int(np.clip(round(float(y0) * scale_y), 0, prior_h - 1))
        px1 = int(np.clip(round(float(x1) * scale_x), px0 + 1, prior_w))
        py1 = int(np.clip(round(float(y1) * scale_y), py0 + 1, prior_h))
        crop = prior[py0:py1, px0:px1]
        crop = crop - crop.min()
        crop = crop / max(float(crop.max()), self.anatomy_guided_masking_epsilon)
        crop_img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
        grid = np.asarray(
            crop_img.resize((patch_grid_size, patch_grid_size), resample=Image.BICUBIC),
            dtype=np.float32,
        )
        grid = np.clip(grid / 255.0, 0, None)
        return grid.astype(np.float32)

    def _finalize_tensor(self, image, *, is_local):
        image = self.to_tensor(image)
        if is_local:
            image = self.local_tensor_noise(image)
        else:
            image = self.global_tensor_noise(image)
        # Frequency-domain augmentation: applied after to_tensor (float32 [0,1]),
        # before normalize, so the frequency statistics are in the raw image space.
        if self.spectral_augmentation is not None:
            image = self.spectral_augmentation(image)
        image = self.normalize(image)
        return image

    def _sample_global_crop(self, image):
        crop, flipped, _ = self._sample_global_crop_with_meta(image)
        return crop, flipped

    def _sample_global_crop_with_meta(self, image, view_index=0):
        width, height = image.size
        if self.global_crop_strategy == "dental_representative_tooth_centered":
            crop, box = self._sample_tooth_centered_global_crop(image, view_index=view_index)
        else:
            top, left, crop_h, crop_w = v2.RandomResizedCrop.get_params(
                image,
                scale=list(self.global_crops_scale),
                ratio=list(self.global_random_crop_ratio),
            )
            crop = image.crop((left, top, left + crop_w, top + crop_h))
            crop = crop.resize((self.global_random_crop.size[1], self.global_random_crop.size[0]), resample=Image.BICUBIC)
            box = (float(left), float(top), float(left + crop_w), float(top + crop_h))
        flipped = random.random() < self.horizontal_flip_p
        if flipped:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        return crop, flipped, box

    def _sample_tooth_centered_global_crop(self, image, view_index=0):
        centers, y_center = self._get_representative_tooth_centers_fast(image)
        width, height = image.size
        if centers:
            center_y = float(np.mean([cy for _, cy in centers]))
        else:
            center_y = y_center
        scale_min, scale_max = self.global_tooth_center_scale
        scale_value = float(random.uniform(scale_min, scale_max))
        crop_side = int(round(min(width * scale_value, height * 0.92)))
        crop_side = int(np.clip(crop_side, min(width, height) * 0.35, min(width, height)))

        view_offsets = (-0.5, 0.5)
        offset = view_offsets[view_index % len(view_offsets)] * self.global_tooth_center_jitter
        center_x = width * float(np.clip(0.5 + offset + random.uniform(-0.5, 0.5) * self.global_tooth_center_jitter, 0.20, 0.80))
        center_y_px = height * float(np.clip(center_y + random.uniform(-0.5, 0.5) * self.global_tooth_center_jitter, 0.35, 0.78))

        x0 = int(np.clip(round(center_x - crop_side / 2), 0, max(width - crop_side, 0)))
        y0 = int(np.clip(round(center_y_px - crop_side / 2), 0, max(height - crop_side, 0)))
        x1 = x0 + crop_side
        y1 = y0 + crop_side
        crop_size = self.global_random_crop.size
        crop = image.crop((x0, y0, x1, y1)).resize((crop_size[1], crop_size[0]), resample=Image.BICUBIC)
        return crop, (float(x0), float(y0), float(x1), float(y1))

    def _sample_hierarchical_local_crops(self, global_bases):
        return self._sample_original_hierarchical_local_crops(global_bases, self.local_crops_number)

    def _sample_hierarchical_2plus4_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        # 2 large context crops: one per global view.
        large_items = []
        for source_global_index, base in enumerate(global_bases):
            large_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=0,
                    count=1,
                    width_range=(0.28, 0.36),
                    height_range=(0.28, 0.38),
                    y_offsets=(-0.02, 0.00, 0.02),
                    existing_boxes=None,
                    iou_threshold=0.24,
                )
            )

        # 2 medium crops: one per global view.
        medium_items = []
        for source_global_index, base in enumerate(global_bases):
            existing_boxes = [item['box'] for item in large_items if item['source_global_index'] == source_global_index]
            medium_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=2,
                    count=1,
                    width_range=(0.20, 0.26),
                    height_range=(0.20, 0.28),
                    y_offsets=(-0.02, 0.00, 0.02),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.10,
                )
            )

        # 4 fine crops: two per global view, avoid overlap with large+medium crops.
        fine_items = []
        for source_global_index, base in enumerate(global_bases):
            existing_boxes = [item['box'] for item in large_items + medium_items if item['source_global_index'] == source_global_index]
            fine_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=1,
                    count=2,
                    width_range=(0.12, 0.18),
                    height_range=(0.14, 0.20),
                    y_offsets=(-0.04, 0.00, 0.04),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.16,
                )
            )

        plan = (
            sorted(large_items, key=lambda item: item['source_global_index'])
            + sorted(medium_items, key=lambda item: item['source_global_index'])
            + sorted(fine_items, key=lambda item: item['source_global_index'])
        )
        for item in plan:
            base = global_bases[item['source_global_index']]
            local_crops.append(self._finalize_tensor(self.local_transfo(item['crop']), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], item['box']))
            local_region_labels.append(item['region_label'])
            local_crop_boxes.append(tuple(float(v) for v in item['box']))
            local_source_global_indices.append(item['source_global_index'])
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_hierarchical_10layer_local_crops(self, image):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        large_items = self._select_stage_crops_for_base(
            image,
            0,
            region_label=0,
            count=2,
            width_range=(0.24, 0.34),
            height_range=(0.24, 0.34),
            y_offsets=(-0.01, 0.01, 0.03),
            existing_boxes=None,
            iou_threshold=0.26,
        )
        medium_items = self._select_stage_crops_for_base(
            image,
            0,
            region_label=2,
            count=2,
            width_range=(0.18, 0.24),
            height_range=(0.18, 0.26),
            y_offsets=(0.00, 0.03, 0.06),
            existing_boxes=[item['box'] for item in large_items],
            iou_threshold=0.18,
        )
        fine_items = self._select_stage_crops_for_base(
            image,
            0,
            region_label=1,
            count=4,
            width_range=(0.10, 0.16),
            height_range=(0.12, 0.18),
            y_offsets=(0.00, 0.03, 0.06, 0.09),
            existing_boxes=[item['box'] for item in large_items + medium_items],
            iou_threshold=0.08,
        )
        lower_items = self._select_stage_crops_for_base(
            image,
            0,
            region_label=3,
            count=2,
            width_range=(0.12, 0.18),
            height_range=(0.14, 0.20),
            y_offsets=(0.10, 0.14, 0.18),
            existing_boxes=[item['box'] for item in large_items + medium_items + fine_items],
            iou_threshold=0.10,
        )

        plan = large_items + medium_items + fine_items + lower_items
        for item in plan:
            local_crops.append(self._finalize_tensor(self.local_transfo(item['crop']), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], item['box']))
            local_region_labels.append(item['region_label'])
            local_crop_boxes.append(tuple(float(v) for v in item['box']))
            local_source_global_indices.append(0)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_hierarchical_12layer_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        large_items = []
        for source_global_index, base in enumerate(global_bases):
            large_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=0,
                    count=1,
                    width_range=(0.28, 0.36),
                    height_range=(0.28, 0.38),
                    y_offsets=(-0.02, 0.00, 0.02),
                    existing_boxes=None,
                    iou_threshold=0.30,
                )
            )

        medium_items = []
        for source_global_index, base in enumerate(global_bases):
            existing_boxes = [item['box'] for item in large_items if item['source_global_index'] == source_global_index]
            medium_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=2,
                    count=1,
                    width_range=(0.20, 0.26),
                    height_range=(0.20, 0.28),
                    y_offsets=(-0.02, 0.00, 0.02),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.22,
                )
            )

        fine_items = []
        lower_items = []
        for source_global_index, base in enumerate(global_bases):
            existing_boxes = [item['box'] for item in large_items + medium_items if item['source_global_index'] == source_global_index]
            fine_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=1,
                    count=3,
                    width_range=(0.12, 0.18),
                    height_range=(0.14, 0.20),
                    y_offsets=(-0.05, -0.02, 0.02, 0.05),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.10,
                )
            )
            lower_existing = existing_boxes + [item['box'] for item in fine_items if item['source_global_index'] == source_global_index]
            lower_items.extend(
                self._select_stage_crops_for_base(
                    base,
                    source_global_index,
                    region_label=3,
                    count=1,
                    width_range=(0.14, 0.20),
                    height_range=(0.16, 0.22),
                    y_offsets=(0.10, 0.14, 0.18),
                    existing_boxes=lower_existing,
                    iou_threshold=0.14,
                )
            )

        plan = (
            sorted(large_items, key=lambda item: item['source_global_index'])
            + sorted(medium_items, key=lambda item: item['source_global_index'])
            + sorted(fine_items, key=lambda item: (item['source_global_index'], item['score']), reverse=False)
            + sorted(lower_items, key=lambda item: item['source_global_index'])
        )
        # Reorder as 2 large + 2 medium + 8 fine/lower.
        plan = (
            sorted(large_items, key=lambda item: item['source_global_index'])
            + sorted(medium_items, key=lambda item: item['source_global_index'])
            + sorted(fine_items[:4], key=lambda item: item['source_global_index'])
            + sorted(fine_items[4:], key=lambda item: item['source_global_index'])
            + sorted(lower_items, key=lambda item: item['source_global_index'])
        )
        for item in plan:
            base = global_bases[item['source_global_index']]
            local_crops.append(self._finalize_tensor(self.local_transfo(item['crop']), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], item['box']))
            local_region_labels.append(item['region_label'])
            local_crop_boxes.append(tuple(float(v) for v in item['box']))
            local_source_global_indices.append(item['source_global_index'])
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_hierarchical_multilevel_local_crops(self, global_bases):
        extra_fine = min(2, max(self.local_crops_number - 1, 0))
        base_count = max(self.local_crops_number - extra_fine, 1)
        (
            local_crops,
            local_side_labels,
            local_region_labels,
            local_crop_boxes,
            local_source_global_indices,
        ) = self._sample_original_hierarchical_local_crops(global_bases, base_count)

        fine_count_per_source = [extra_fine // len(global_bases) for _ in global_bases]
        for i in range(extra_fine % len(global_bases)):
            fine_count_per_source[i] += 1

        for source_global_index, base in enumerate(global_bases):
            count = fine_count_per_source[source_global_index]
            if count <= 0:
                continue
            existing_boxes = [
                box for box, src in zip(local_crop_boxes, local_source_global_indices) if src == source_global_index
            ]
            fine_items = self._select_stage_crops_for_base(
                base,
                source_global_index,
                region_label=1,
                count=count,
                width_range=(0.12, 0.18),
                height_range=(0.14, 0.20),
                y_offsets=(-0.03, 0.00, 0.03),
                existing_boxes=existing_boxes,
                iou_threshold=0.20,
            )
            for item in fine_items:
                local_crops.append(self._finalize_tensor(self.local_transfo(item['crop']), is_local=True))
                local_side_labels.append(self._encode_side_label(base.size[0], item['box']))
                local_region_labels.append(item['region_label'])
                local_crop_boxes.append(tuple(float(v) for v in item['box']))
                local_source_global_indices.append(item['source_global_index'])
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_original_hierarchical_local_crops(self, global_bases, total_local_crops):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        n_quadrant = int(round(total_local_crops * self.local_quadrant_ratio))
        n_transition = int(round(total_local_crops * self.hierarchical_transition_ratio))
        n_transition = min(n_transition, max(total_local_crops - n_quadrant, 0))
        for index in range(total_local_crops):
            source_global_index = index % len(global_bases)
            base = global_bases[source_global_index]
            if index < n_quadrant:
                crop, box = self._sample_quadrant_crop(base)
                region_label = 0
            elif index < n_quadrant + n_transition:
                crop, box = self._sample_transition_crop(base)
                region_label = 2
            else:
                crop, box = self._sample_arch_patch(base)
                region_label = 1
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(source_global_index)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_original_compact_hierarchical_local_crops(self, image, total_local_crops):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        n_quadrant = int(round(total_local_crops * self.local_quadrant_ratio))
        n_transition = int(round(total_local_crops * self.hierarchical_transition_ratio))
        n_transition = min(n_transition, max(total_local_crops - n_quadrant, 0))
        n_arch = max(total_local_crops - n_quadrant - n_transition, 0)

        _, _, band = self._build_tooth_focus_maps(image)
        band = self._stabilize_panorama_focus_band(band)
        x_left, x_right, y_center = band
        band_w = max(x_right - x_left, 0.10)
        upper_y = float(np.clip(y_center - 0.14, 0.36, 0.56))
        lower_y = float(np.clip(y_center + 0.09, 0.52, 0.80))
        quadrant_centers = [
            (float(np.clip(x_left + 0.32 * band_w, 0.10, 0.90)), upper_y),
            (float(np.clip(x_left + 0.68 * band_w, 0.10, 0.90)), upper_y),
            (float(np.clip(x_left + 0.30 * band_w, 0.10, 0.90)), float(np.clip(lower_y - 0.01, 0.48, 0.78)) ), 
            #
            (float(np.clip(x_left + 0.68 * band_w, 0.10, 0.90)), lower_y),
        ]
        arch_centers = [
            (float(np.clip(x_left + 0.20 * band_w, 0.10, 0.90)), float(np.clip(y_center - 0.03, 0.46, 0.76))),
            (float(np.clip(x_left + 0.38 * band_w, 0.10, 0.90)), float(np.clip(y_center - 0.02, 0.48, 0.78))),
            (float(np.clip(x_left + 0.50 * band_w, 0.10, 0.90)), float(np.clip(y_center + 0.08, 0.52, 0.82))),
            (float(np.clip(x_left + 0.62 * band_w, 0.10, 0.90)), float(np.clip(y_center - 0.01, 0.48, 0.78))),
            (float(np.clip(x_left + 0.88 * band_w, 0.10, 0.90)), float(np.clip(y_center + 0.01, 0.48, 0.78))),
        ]
        transition_centers = [
            (float(np.clip(x_left + 0.35 * band_w, 0.10, 0.90)), float(np.clip(y_center - 0.04, 0.44, 0.76))),
            (float(np.clip(x_left + 0.65 * band_w, 0.10, 0.90)), float(np.clip(y_center + 0.04, 0.44, 0.76))),
        ]

        existing_boxes = []
        for index in range(total_local_crops):
            if index < n_quadrant:
                center = quadrant_centers[index % len(quadrant_centers)]
                crop, box = self._sample_toothband_guided_crop(
                    image,
                    center=center,
                    width_range=(0.30, 0.38),
                    height_range=(0.24, 0.32),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.28,
                )
                region_label = 0
            elif index < n_quadrant + n_transition:
                center = transition_centers[(index - n_quadrant) % len(transition_centers)]
                crop, box = self._sample_toothband_guided_crop(
                    image,
                    center=center,
                    width_range=(0.16, 0.22),
                    height_range=(0.14, 0.20),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.14,
                )
                region_label = 2
            else:
                arch_index = (index - n_quadrant - n_transition) % len(arch_centers)
                center = arch_centers[arch_index]
                crop, box = self._sample_toothband_guided_crop(
                    image,
                    center=center,
                    width_range=(0.12, 0.18),
                    height_range=(0.11, 0.17) if arch_index != 2 else (0.12, 0.19),
                    existing_boxes=existing_boxes,
                    iou_threshold=0.08,
                )
                region_label = 1
            existing_boxes.append(box)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(0)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _stabilize_panorama_focus_band(self, band):
        """Keep saliency-based panoramic tooth bands from collapsing to one side."""
        x_left, x_right, y_center = band
        width = max(x_right - x_left, 0.0)
        center = 0.5 * (x_left + x_right)

        if width < 0.62 or center < 0.36 or center > 0.64:
            center = 0.5
            width = max(width, 0.68)
        else:
            center = 0.70 * center + 0.30 * 0.5
            width = max(width, 0.62)

        x_left = max(0.10, center - width * 0.5)
        x_right = min(0.90, center + width * 0.5)
        if x_right - x_left < 0.62:
            if x_left <= 0.10:
                x_right = min(0.90, x_left + 0.62)
            elif x_right >= 0.90:
                x_left = max(0.10, x_right - 0.62)
        return x_left, x_right, y_center

    def _sample_original_compact_hierarchical_plus2_local_crops(self, image):
        (
            local_crops,
            local_side_labels,
            local_region_labels,
            local_crop_boxes,
            local_source_global_indices,
        ) = self._sample_original_compact_hierarchical_local_crops(image, 8)
        existing_boxes = [tuple(float(v) for v in box) for box in local_crop_boxes]
        _, _, band = self._build_tooth_focus_maps(image)
        x_left, x_right, y_center = band
        random_centers = [
            (random.uniform(x_left, x_right), float(np.clip(y_center + random.uniform(-0.03, 0.10), 0.48, 0.82))),
            (random.uniform(x_left, x_right), float(np.clip(y_center + random.uniform(-0.03, 0.10), 0.48, 0.82))),
        ]
        for center in random_centers:
            crop, box = self._sample_toothband_guided_crop(
                image,
                center=center,
                width_range=(0.09, 0.14),
                height_range=(0.09, 0.14),
                existing_boxes=existing_boxes,
                iou_threshold=0.06,
            )
            existing_boxes.append(box)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], box))
            local_region_labels.append(3)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(0)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_original_compact_hierarchical_multiscale_local_crops(self, image):
        (
            local_crops,
            local_side_labels,
            local_region_labels,
            local_crop_boxes,
            local_source_global_indices,
        ) = self._sample_original_compact_hierarchical_local_crops(image, min(self.local_crops_number, 8))
        extra_count = max(self.local_crops_number - len(local_crops), 0)
        if extra_count <= 0:
            return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

        existing_boxes = [tuple(float(v) for v in box) for box in local_crop_boxes]
        quadrant_count = min(extra_count, 4, len(local_crop_boxes))
        if quadrant_count > 0:
            parent_boxes = [local_crop_boxes[i] for i in range(quadrant_count)]
        else:
            parent_boxes = list(local_crop_boxes[:extra_count])
        for parent_box in parent_boxes:
            crop, box = self._sample_subcrop_within_parent(
                image,
                parent_box,
                existing_boxes=existing_boxes,
                scale_range=(0.34, 0.48),
                iou_threshold=0.08,
            )
            existing_boxes.append(box)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], box))
            local_region_labels.append(3)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(0)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_original_compact_hierarchical_multiscale_fast_local_crops(self, image):
        return self._sample_band_aware_fast_local_crops(image, self.local_crops_number)

    def _representative_tooth_centers_from_band(self, band):
        x_left, x_right, y_center = self._stabilize_panorama_focus_band(band)
        band_w = max(x_right - x_left, 0.10)
        x_mid = 0.5 * (x_left + x_right)
        compressed_band_w = 0.58 * band_w
        central_left = float(np.clip(x_mid - 0.5 * compressed_band_w, 0.16, 0.84))
        central_right = float(np.clip(x_mid + 0.5 * compressed_band_w, 0.16, 0.84))
        central_band_w = max(central_right - central_left, 0.10)

        upper_y = float(np.clip(y_center - 0.10, 0.36, 0.58))
        lower_y = float(np.clip(y_center + 0.08, 0.48, 0.80))
        upper_rel_xs = (0.12, 0.38, 0.62, 0.88)
        lower_rel_xs = (0.14, 0.40, 0.60, 0.86)
        centers = [
            (central_left + rel_x * central_band_w, upper_y) for rel_x in upper_rel_xs
        ] + [
            (central_left + rel_x * central_band_w, lower_y) for rel_x in lower_rel_xs
        ]
        return centers, y_center

    def _get_representative_tooth_centers(self, image):
        _, _, band = self._build_tooth_focus_maps(image)
        return self._representative_tooth_centers_from_band(band)

    def _get_representative_tooth_centers_fast(self, image):
        max_work_side = max(self.global_crops_size * 2, self.local_crops_size * 4)
        band = self._estimate_tooth_focus_band_fast(image, max_side=max_work_side)
        return self._representative_tooth_centers_from_band(band)

    def _estimate_tooth_focus_band_fast(self, image, max_side=512):
        width, height = image.size
        scale = min(1.0, float(max_side) / max(width, height, 1))
        if scale < 1.0:
            work_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(work_size, resample=Image.BILINEAR)

        gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
        grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge = grad_x + 0.75 * grad_y

        h, w = edge.shape
        y_coords = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
        row_prior = np.exp(-0.5 * ((y_coords - 0.64) / 0.16) ** 2)
        intensity_weight = np.clip((gray - 0.18) / 0.52, 0.0, 1.0)
        saliency = edge * (0.35 + 0.65 * intensity_weight) * row_prior[:, None]

        kernel = np.ones(9, dtype=np.float32) / 9.0
        row_slice = saliency[int(h * 0.36): int(h * 0.90)]
        col_energy = row_slice.mean(axis=0)
        col_energy = np.convolve(col_energy, kernel, mode="same")
        threshold = 0.48 * max(float(col_energy.max()), 1e-6)
        xs = np.where(col_energy >= threshold)[0]
        if xs.size == 0:
            x_left, x_right = 0.18, 0.82
        else:
            x_left = max(0.10, float(xs[0]) / max(w - 1, 1) - 0.03)
            x_right = min(0.90, float(xs[-1]) / max(w - 1, 1) + 0.03)

        x0 = int(w * x_left)
        x1 = max(x0 + 1, int(w * x_right))
        col_slice = saliency[:, x0:x1]
        row_energy = col_slice.mean(axis=1)
        row_energy = np.convolve(row_energy, kernel, mode="same")
        row_threshold = 0.58 * max(float(row_energy.max()), 1e-6)
        ys = np.where(row_energy >= row_threshold)[0]
        if ys.size == 0:
            y_center = 0.62
        else:
            y_center = float(np.clip(ys.mean() / max(h - 1, 1), 0.50, 0.78))
        return x_left, x_right, y_center

    def _sample_representative_tooth_aware_local_crops(self, image):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        centers, y_center = self._get_representative_tooth_centers(image)
        gray, saliency, _ = self._build_tooth_focus_maps(image)
        width, height = image.size
        image_area = float(width * height)
        log_aspect_min = np.log(self.local_random_crop_ratio[0])
        log_aspect_max = np.log(self.local_random_crop_ratio[1])
        total_local_crops = min(self.local_crops_number, len(centers))

        for center in centers[:total_local_crops]:
            best_box = None
            best_crop = None
            best_score = None
            existing_boxes = [tuple(float(v) for v in box) for box in local_crop_boxes]
            for _ in range(self.representative_tooth_aware_attempts):
                area_scale = random.uniform(*self.local_crops_scale)
                aspect_ratio = float(np.exp(random.uniform(log_aspect_min, log_aspect_max)))
                crop_w = int(round(np.sqrt(image_area * area_scale * aspect_ratio)))
                crop_h = int(round(np.sqrt(image_area * area_scale / aspect_ratio)))
                crop_w = int(np.clip(crop_w, self.local_crops_size, width))
                crop_h = int(np.clip(crop_h, self.local_crops_size, height))

                center_x = width * center[0] + random.uniform(-0.015, 0.015) * width
                center_y = height * center[1] + random.uniform(-0.015, 0.015) * height
                crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

                overlaps = [self._box_iou(box, prev_box) for prev_box in existing_boxes]
                overlap_penalty = sum(overlaps)
                max_overlap = max(overlaps) if overlaps else 0.0
                score = self._score_crop_structure(image, box, gray=gray, saliency=saliency, focus_y=y_center)
                score -= 0.45 * overlap_penalty
                if max_overlap > 0.30:
                    score -= 0.35

                x0, y0, x1, y1 = box
                height_ratio = (y1 - y0) / max(height, 1)
                width_ratio = (x1 - x0) / max(width, 1)
                box_center_y = 0.5 * (y0 + y1) / max(height, 1)
                top_overflow = max(0.0, 0.18 - (y0 / max(height, 1)))
                bottom_overflow = max(0.0, (y1 / max(height, 1)) - 0.92)
                vertical_center_error = abs(box_center_y - center[1])

                score -= 2.2 * max(0.0, height_ratio - 0.56)
                score -= 0.8 * max(0.0, width_ratio - 0.46)
                score -= 4.5 * (top_overflow + bottom_overflow)
                score -= 0.9 * vertical_center_error

                if best_box is None or score > best_score:
                    best_crop = crop
                    best_box = box
                    best_score = score

            if best_box is None or best_crop is None:
                best_crop, best_box = self._crop_from_center(
                    image,
                    width * center[0],
                    height * center[1],
                    self.local_crops_size,
                    self.local_crops_size,
                )
            local_crops.append(self._finalize_tensor(self.local_transfo(best_crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], best_box))
            local_region_labels.append(1)
            local_crop_boxes.append(tuple(float(v) for v in best_box))
            local_source_global_indices.append(0)

        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_representative_tooth_direct_local_crops(
        self, image, image_path=None, center_jitter=0.0, random_scale=False
    ):
        centers = self._get_cached_representative_tooth_centers(image_path)
        if centers is None:
            centers, _ = self._get_representative_tooth_centers_fast(image)
        return self._sample_fixed_center_local_crops(
            image, centers, center_jitter=center_jitter, random_scale=random_scale
        )

    def _sample_representative_tooth_static_direct_local_crops(self, image):
        centers = (
            (0.24, 0.52),
            (0.41, 0.52),
            (0.59, 0.52),
            (0.76, 0.52),
            (0.25, 0.70),
            (0.42, 0.70),
            (0.58, 0.70),
            (0.75, 0.70),
        )
        return self._sample_fixed_center_local_crops(image, centers)

    def _sample_representative_tooth_adaptive_direct_local_crops(self, image):
        centers, _ = self._get_representative_tooth_centers(image)
        return self._sample_fixed_center_local_crops(image, centers)

    def _sample_tcc_legacy_local_crops(self, image, image_path=None):
        """TCC as it was before the n-TCC scale fix. Frozen for ablation -- do not modify.

        Differences from _sample_fixed_center_local_crops (n-TCC):
          crop area is a fraction of W*H rather than H*H, crops are taken from a
          downscaled working copy, the crop side is floored at local_crops_size, and
          the local branch gets no horizontal flip. Reached via
          local_crop_strategy: tcc_legacy, with direct_center_jitter / direct_random_scale
          selecting the stochastic or deterministic legacy variant.
        """
        centers = self._get_cached_representative_tooth_centers(image_path)
        if centers is None:
            centers, _ = self._get_representative_tooth_centers_fast(image)

        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        center_jitter = self.direct_center_jitter
        random_scale = self.direct_random_scale

        max_work_side = max(self.global_crops_size * 2, self.local_crops_size * 4)
        original_width, original_height = image.size
        max_original_side = max(original_width, original_height)
        if random_scale:
            max_work_side = max_original_side
        work_to_original_scale = 1.0
        if max_original_side > max_work_side:
            scale = float(max_work_side) / float(max_original_side)
            work_size = (
                max(self.local_crops_size, int(round(original_width * scale))),
                max(self.local_crops_size, int(round(original_height * scale))),
            )
            image = image.resize(work_size, resample=Image.BICUBIC)
            work_to_original_scale = 1.0 / scale

        width, height = image.size
        image_area = float(width * height)
        fixed_area_scale = float(np.sqrt(self.local_crops_scale[0] * self.local_crops_scale[1]))
        log_ratio_min = float(np.log(self.local_random_crop_ratio[0]))
        log_ratio_max = float(np.log(self.local_random_crop_ratio[1]))

        total_local_crops = min(self.local_crops_number, len(centers))
        for center in centers[:total_local_crops]:
            if random_scale:
                area_scale = float(random.uniform(self.local_crops_scale[0], self.local_crops_scale[1]))
                ratio = float(np.exp(random.uniform(log_ratio_min, log_ratio_max)))
            else:
                area_scale = fixed_area_scale
                ratio = 1.0
            target_area = image_area * area_scale
            crop_w = int(np.clip(round(np.sqrt(target_area * ratio)), self.local_crops_size, width))
            crop_h = int(np.clip(round(np.sqrt(target_area / ratio)), self.local_crops_size, height))

            center_x = width * center[0]
            center_y = height * center[1]
            if center_jitter > 0.0:
                center_x += random.uniform(-center_jitter, center_jitter) * width
                center_y += random.uniform(-center_jitter, center_jitter) * height
            crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)
            original_box = tuple(float(v) * work_to_original_scale for v in box)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(original_width, original_box))
            local_region_labels.append(1)
            local_crop_boxes.append(original_box)
            local_source_global_indices.append(0)

        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_fixed_center_local_crops(self, image, centers, *, center_jitter=0.0, random_scale=False):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        # Crop straight from the full-resolution panorama, exactly like the official
        # random strategy. Downscaling to a working image first would (a) floor the
        # reachable crop area at local_crops_size^2 / work_area, truncating the low end of
        # local_crops_scale, (b) put a second resampling step in the pipeline, confounding
        # an n-TCC vs random ablation with a resolution difference, and (c) cost more, not
        # less: measured on a 2774x1504 panorama, 14.8 ms cropping directly vs 41.7 ms via
        # a 1280 px working copy, because resizing touches every pixel of the panorama
        # while the eight crops together touch only part of it.
        width, height = image.size
        # n-TCC: anchor the crop area to H^2, not to W*H. Panoramic radiographs are ~2:1,
        # so a square crop sized from full image area comes out sqrt(W/H) ~ 1.41x taller
        # than the DINOv3 local_crops_scale semantics intend -- up to 0.80 H, at which
        # point _crop_from_center clamps the box back inside the image and drags the lower
        # tooth row's crops off their anchors toward the upper row. Anchored to H^2,
        # side = sqrt(area_scale) * H and every crop stays on its anchor.
        reference_area = float(height * height)
        # Deterministic fallback: geometric mean of the configured area scale.
        fixed_area_scale = float(np.sqrt(self.local_crops_scale[0] * self.local_crops_scale[1]))
        log_ratio_min = float(np.log(self.local_random_crop_ratio[0]))
        log_ratio_max = float(np.log(self.local_random_crop_ratio[1]))

        total_local_crops = min(self.local_crops_number, len(centers))
        for center in centers[:total_local_crops]:
            if random_scale:
                # Sample area uniformly (same semantics as RandomResizedCrop) plus an
                # aspect-ratio draw, so the crop geometry differs on every epoch.
                area_scale = float(random.uniform(self.local_crops_scale[0], self.local_crops_scale[1]))
                ratio = float(np.exp(random.uniform(log_ratio_min, log_ratio_max)))
            else:
                area_scale = fixed_area_scale
                ratio = 1.0
            target_area = reference_area * area_scale
            crop_w = int(np.clip(round(np.sqrt(target_area * ratio)), _LOCAL_CROP_MIN_PIXELS, width))
            crop_h = int(np.clip(round(np.sqrt(target_area / ratio)), _LOCAL_CROP_MIN_PIXELS, height))

            center_x = width * center[0]
            center_y = height * center[1]
            if center_jitter > 0.0:
                center_x += random.uniform(-center_jitter, center_jitter) * width
                center_y += random.uniform(-center_jitter, center_jitter) * height
            crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)
            # Left-right mirroring of the dental arch is anatomically valid and is the
            # cheapest remaining source of geometric entropy for the local branch:
            # MedicalImageAugmentation is built with horizontal_flip=0.0 and the config's
            # horizontal_flips only reaches global crops. local_side_labels are collated
            # but consumed by no loss, so the label stays in unflipped coordinates.
            if random.random() < self.horizontal_flip_p:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            source_box = tuple(float(v) for v in box)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(width, source_box))
            local_region_labels.append(1)
            local_crop_boxes.append(source_box)
            local_source_global_indices.append(0)

        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_hybrid_random_band_local_crops(self, image):
        random_count = min(4, self.local_crops_number)
        band_count = max(self.local_crops_number - random_count, 0)

        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        for _ in range(random_count):
            crop, box, flipped = self._sample_official_random_local_crop_with_box(image)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], box))
            local_region_labels.append(0)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(0)

        if band_count > 0:
            (
                band_local_crops,
                band_local_side_labels,
                band_local_region_labels,
                band_local_crop_boxes,
                band_local_source_global_indices,
            ) = self._sample_band_aware_fast_local_crops(image, band_count)
            local_crops.extend(band_local_crops)
            local_side_labels.extend(band_local_side_labels)
            local_region_labels.extend(band_local_region_labels)
            local_crop_boxes.extend(band_local_crop_boxes)
            local_source_global_indices.extend(band_local_source_global_indices)

        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_band_aware_fast_local_crops(self, image, total_local_crops):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []

        _, _, band = self._build_tooth_focus_maps(image)
        x_left, x_right, y_center = self._stabilize_panorama_focus_band(band)
        band_w = max(x_right - x_left, 0.10)
        width, height = image.size

        upper_y = float(np.clip(y_center - self.band_aware_fast_upper_center_offset, 0.36, 0.58))
        lower_y = float(np.clip(y_center + self.band_aware_fast_lower_center_offset, 0.48, 0.82))

        # Semi-random band-aware layout: keep crops around the dental arch band,
        # but sample positions inside broad bins so the 8 locals stay diverse.
        upper_bins = [(0.06, 0.28), (0.22, 0.46), (0.54, 0.78), (0.72, 0.94)]
        lower_bins = [(0.08, 0.32), (0.28, 0.52), (0.48, 0.72), (0.68, 0.92)]
        random.shuffle(upper_bins)
        random.shuffle(lower_bins)

        upper_specs = []
        for x_bin in upper_bins:
            rel_x = random.uniform(*x_bin)
            rel_y = float(np.clip(upper_y + random.uniform(-0.05, 0.03), 0.32, 0.64))
            upper_specs.append(((float(np.clip(x_left + rel_x * band_w, 0.08, 0.92)), rel_y), 0))
        lower_specs = []
        for x_bin in lower_bins:
            rel_x = random.uniform(*x_bin)
            rel_y = float(np.clip(lower_y + random.uniform(-0.03, 0.05), 0.42, 0.86))
            lower_specs.append(((float(np.clip(x_left + rel_x * band_w, 0.08, 0.92)), rel_y), 1))

        center_specs = []
        for up, low in zip(upper_specs, lower_specs):
            center_specs.extend([up, low])

        if self.band_aware_fast_use_official_area_scale:
            min_side_ratio = float(np.sqrt(max(self.local_crops_scale[0], 1e-6)))
            max_side_ratio = float(np.sqrt(max(self.local_crops_scale[1], 1e-6)))
        else:
            min_side_ratio = self.band_aware_fast_crop_min_ratio
            max_side_ratio = self.band_aware_fast_crop_max_ratio

        min_side = max(self.local_crops_size, int(min(width, height) * min_side_ratio))
        max_side = min(min(width, height), int(min(width, height) * max_side_ratio))
        max_side = max(min_side, max_side)

        for index in range(total_local_crops):
            center, region_label = center_specs[index % len(center_specs)]
            center_x = width * center[0]
            center_y = height * center[1]
            crop_side = int(np.clip(random.uniform(min_side, max_side), self.local_crops_size, min(width, height)))
            crop_w = crop_side
            crop_h = crop_side
            crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(image.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(0)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_official_random_local_crop_with_box(self, image):
        top, left, height, width = v2.RandomResizedCrop.get_params(
            image,
            scale=self.local_crops_scale,
            ratio=self.local_random_crop_ratio,
        )
        box = (left, top, left + width, top + height)
        crop = image.crop(box).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        flipped = random.random() < self.horizontal_flip_p
        if flipped:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        return crop, box, flipped

    def _sample_anatomy_guided_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        n_context = max(1, int(round(self.local_crops_number * self.local_quadrant_ratio)))
        for index in range(self.local_crops_number):
            source_global_index = index % len(global_bases)
            base = global_bases[source_global_index]
            if index < n_context:
                crop, box = self._sample_quadrant_crop(base)
                region_label = 0
            else:
                crop, box = self._sample_anatomy_patch(base)
                region_label = 1
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(source_global_index)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices


    def _sample_balanced_hierarchical_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        n_quadrant = int(round(self.local_crops_number * self.local_quadrant_ratio))
        n_transition = int(round(self.local_crops_number * self.hierarchical_transition_ratio))
        n_transition = min(n_transition, max(self.local_crops_number - n_quadrant, 0))
        quadrant_centers = [(0.28, 0.34), (0.72, 0.34), (0.28, 0.66), (0.72, 0.66)]
        arch_centers = [(0.22, 0.54), (0.42, 0.48), (0.58, 0.52), (0.78, 0.58)]
        transition_centers = [(0.36, 0.44), (0.64, 0.44), (0.36, 0.60), (0.64, 0.60)]
        for index in range(self.local_crops_number):
            source_global_index = index % len(global_bases)
            base = global_bases[source_global_index]
            slot_index = index // len(global_bases)
            if index < n_quadrant:
                center = quadrant_centers[slot_index % len(quadrant_centers)]
                crop, box = self._sample_balanced_quadrant_crop(base, center)
                region_label = 0
            elif index < n_quadrant + n_transition:
                center = transition_centers[(index - n_quadrant) // len(global_bases) % len(transition_centers)]
                crop, box = self._sample_balanced_transition_crop(base, center)
                region_label = 2
            else:
                center = arch_centers[(index - n_quadrant - n_transition) // len(global_bases) % len(arch_centers)]
                crop, box = self._sample_balanced_arch_patch(base, center)
                region_label = 1
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(source_global_index)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _sample_compact_hierarchical_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        n_quadrant = int(round(self.local_crops_number * self.local_quadrant_ratio))
        n_transition = int(round(self.local_crops_number * self.hierarchical_transition_ratio))
        n_transition = min(n_transition, max(self.local_crops_number - n_quadrant, 0))
        for index in range(self.local_crops_number):
            source_global_index = index % len(global_bases)
            base = global_bases[source_global_index]
            if index < n_quadrant:
                crop, box = self._sample_compact_quadrant_crop(base)
                region_label = 0
            elif index < n_quadrant + n_transition:
                crop, box = self._sample_compact_transition_crop(base)
                region_label = 2
            else:
                crop, box = self._sample_compact_arch_patch(base)
                region_label = 1
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(source_global_index)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices


    def _build_tooth_focus_maps(self, image):
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
        grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge = grad_x + 0.75 * grad_y

        h, w = edge.shape
        y_coords = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
        row_prior = np.exp(-0.5 * ((y_coords - 0.64) / 0.16) ** 2)
        intensity_weight = np.clip((gray - 0.18) / 0.52, 0.0, 1.0)
        saliency = edge * (0.35 + 0.65 * intensity_weight) * row_prior[:, None]

        kernel = np.ones(9, dtype=np.float32) / 9.0
        row_slice = saliency[int(h * 0.36): int(h * 0.90)]
        col_energy = row_slice.mean(axis=0)
        col_energy = np.convolve(col_energy, kernel, mode='same')
        threshold = 0.48 * max(float(col_energy.max()), 1e-6)
        xs = np.where(col_energy >= threshold)[0]
        if xs.size == 0:
            x_left, x_right = 0.18, 0.82
        else:
            x_left = max(0.10, float(xs[0]) / max(w - 1, 1) - 0.03)
            x_right = min(0.90, float(xs[-1]) / max(w - 1, 1) + 0.03)

        x0 = int(w * x_left)
        x1 = max(x0 + 1, int(w * x_right))
        col_slice = saliency[:, x0:x1]
        row_energy = col_slice.mean(axis=1)
        row_energy = np.convolve(row_energy, kernel, mode='same')
        row_threshold = 0.58 * max(float(row_energy.max()), 1e-6)
        ys = np.where(row_energy >= row_threshold)[0]
        if ys.size == 0:
            y_center = 0.62
        else:
            y_center = float(np.clip(ys.mean() / max(h - 1, 1), 0.50, 0.78))

        saliency = saliency / max(float(saliency.max()), 1e-6)
        return gray, saliency, (x_left, x_right, y_center)

    def _estimate_tooth_focus_band(self, image):
        _, _, band = self._build_tooth_focus_maps(image)
        return band

    def _sample_band_guided_crop(
        self,
        image,
        band,
        *,
        x_slot,
        num_slots,
        width_range,
        height_range,
        y_offset_choices,
        min_crop,
    ):
        gray, saliency, _ = self._build_tooth_focus_maps(image)
        width, height = image.size
        x_left, x_right, y_center = band
        x_slots = np.linspace(x_left, x_right, num=max(num_slots, 2))
        base_center_x = x_slots[min(x_slot, len(x_slots) - 1)] * width
        best = None
        best_score = None
        for _ in range(8):
            center_x = base_center_x + random.uniform(-0.03, 0.03) * width
            center_y = y_center * height + random.choice(y_offset_choices) * height + random.uniform(-0.02, 0.02) * height
            crop_w = int(np.clip(width * random.uniform(*width_range), min_crop, width))
            crop_h = int(np.clip(height * random.uniform(*height_range), min_crop, height))
            crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)
            score = self._score_crop_structure(image, box, gray=gray, saliency=saliency, focus_y=y_center)
            if best is None or score > best_score:
                best = (crop, box)
                best_score = score
        return best

    def _score_crop_structure(self, image, box, *, gray=None, saliency=None, focus_y=0.62):
        if gray is None or saliency is None:
            gray, saliency, (_, _, focus_y) = self._build_tooth_focus_maps(image)
        x0, y0, x1, y1 = [int(v) for v in box]
        patch = gray[y0:y1, x0:x1]
        saliency_patch = saliency[y0:y1, x0:x1]
        if patch.size == 0 or saliency_patch.size == 0:
            return -1.0
        grad_x = np.abs(np.diff(patch, axis=1, prepend=patch[:, :1]))
        grad_y = np.abs(np.diff(patch, axis=0, prepend=patch[:1, :]))
        edge_strength = float((grad_x + grad_y).mean())
        intensity_std = float(patch.std())
        saliency_mean = float(saliency_patch.mean())
        tooth_coverage = float((saliency_patch > 0.45).mean())
        center_weight = 1.0 - abs(((y0 + y1) * 0.5 / max(gray.shape[0], 1)) - focus_y)
        return edge_strength + 0.20 * intensity_std + 0.70 * saliency_mean + 0.35 * tooth_coverage + 0.08 * center_weight

    def _select_stage_crops_for_base(
        self,
        base,
        source_global_index,
        *,
        region_label,
        count,
        width_range,
        height_range,
        y_offsets,
        existing_boxes=None,
        iou_threshold=0.35,
        soft_iou_weight=0.45,
        diversity_weight=0.08,
    ):
        gray, saliency, band = self._build_tooth_focus_maps(base)
        candidates = []
        width, height = base.size
        min_crop = max(self.local_crops_size // 2, 32)
        x_left, x_right, y_center = band
        slot_centers = np.linspace(x_left, x_right, num=max(count * 4, 8))
        for slot_x in slot_centers:
            for _ in range(10):
                center_x = slot_x * width + random.uniform(-0.03, 0.03) * width
                center_y = y_center * height + random.choice(y_offsets) * height + random.uniform(-0.02, 0.02) * height
                crop_w = int(np.clip(width * random.uniform(*width_range), min_crop, width))
                crop_h = int(np.clip(height * random.uniform(*height_range), min_crop, height))
                crop, box = self._crop_from_center(base, center_x, center_y, crop_w, crop_h)
                base_score = self._score_crop_structure(base, box, gray=gray, saliency=saliency, focus_y=y_center)
                edge_margin = min(box[0], width - box[2], box[1], height - box[3]) / max(min(width, height), 1)
                candidates.append({
                    'crop': crop,
                    'box': box,
                    'base_score': base_score + 0.05 * edge_margin,
                    'region_label': region_label,
                    'source_global_index': source_global_index,
                })

        existing_boxes = existing_boxes or []
        selected = []
        remaining = list(candidates)
        hard_cap = max(iou_threshold, 0.18)

        while len(selected) < count and remaining:
            best_idx = None
            best_score = None
            for idx, item in enumerate(remaining):
                overlaps = [self._box_iou(item['box'], prev['box']) for prev in selected]
                overlaps.extend(self._box_iou(item['box'], prev_box) for prev_box in existing_boxes)
                max_overlap = max(overlaps) if overlaps else 0.0
                if max_overlap > hard_cap:
                    continue
                overlap_penalty = sum(overlaps)
                center_x = 0.5 * (item['box'][0] + item['box'][2]) / max(width, 1)
                prev_centers = [0.5 * (p['box'][0] + p['box'][2]) / max(width, 1) for p in selected]
                diversity_bonus = min(abs(center_x - prev_cx) for prev_cx in prev_centers) if prev_centers else 0.0
                final_score = item['base_score'] - soft_iou_weight * overlap_penalty + diversity_weight * diversity_bonus
                if best_score is None or final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            if best_idx is None:
                hard_cap = min(0.45, hard_cap + 0.05)
                if hard_cap >= 0.45:
                    break
                continue
            chosen = remaining.pop(best_idx)
            chosen['score'] = best_score
            selected.append(chosen)

        remaining.sort(key=lambda item: item['base_score'], reverse=True)
        while len(selected) < count and remaining:
            fallback = remaining.pop(0)
            fallback['score'] = fallback['base_score']
            selected.append(fallback)
        return selected

    def _box_iou(self, box_a, box_b):
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _sample_foreground_dental_mixed_local_crops(self, global_bases):
        local_crops = []
        local_side_labels = []
        local_region_labels = []
        local_crop_boxes = []
        local_source_global_indices = []
        crop_types = self._build_foreground_dental_crop_plan()
        grayscale_maps = [self._get_grayscale_map(base) for base in global_bases]

        for index, crop_type in enumerate(crop_types):
            source_global_index = index % len(global_bases)
            base = global_bases[source_global_index]
            gray_map = grayscale_maps[source_global_index]
            crop, box, region_label = self._sample_mixed_policy_crop(base, gray_map, crop_type)
            local_crops.append(self._finalize_tensor(self.local_transfo(crop), is_local=True))
            local_side_labels.append(self._encode_side_label(base.size[0], box))
            local_region_labels.append(region_label)
            local_crop_boxes.append(tuple(float(v) for v in box))
            local_source_global_indices.append(source_global_index)
        return local_crops, local_side_labels, local_region_labels, local_crop_boxes, local_source_global_indices

    def _build_foreground_dental_crop_plan(self):
        plan = (
            ["random_foreground"] * max(self.local_policy["random_foreground_number"], 0)
            + ["fine_dental"] * max(self.local_policy["fine_dental_number"], 0)
            + ["anchor_dental"] * max(self.local_policy["anchor_dental_number"], 0)
        )
        if len(plan) < self.local_crops_number:
            plan.extend(["random_foreground"] * (self.local_crops_number - len(plan)))
        elif len(plan) > self.local_crops_number:
            plan = plan[: self.local_crops_number]
        random.shuffle(plan)
        return plan

    def _sample_mixed_policy_crop(self, image, gray_map, crop_type):
        if crop_type == "fine_dental":
            crop, box = self._sample_foreground_aware_random_crop(
                image,
                gray_map,
                scale_range=self.local_policy["fine_dental_scale"],
            )
            return crop, box, 1
        if crop_type == "anchor_dental":
            crop, box = self._sample_anchor_dental_crop(image, gray_map)
            return crop, box, 2
        crop, box = self._sample_foreground_aware_random_crop(
            image,
            gray_map,
            scale_range=self.local_policy["random_foreground_scale"],
        )
        return crop, box, 0

    def _sample_foreground_aware_random_crop(self, image, gray_map, scale_range):
        box = self._sample_box_with_foreground_constraint(image, gray_map, scale_range=scale_range)
        crop = image.crop(box).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, box

    def _sample_anchor_dental_crop(self, image, gray_map):
        max_attempts = max(self.local_policy["max_resample_attempts"], 1)
        last_box = None
        for _ in range(max_attempts):
            anchor_x, anchor_y = random.choice(self.anchor_regions)
            center_x = np.clip(
                anchor_x + random.uniform(-self.local_policy["center_jitter"], self.local_policy["center_jitter"]),
                0.0,
                1.0,
            )
            center_y = np.clip(
                anchor_y + random.uniform(-self.local_policy["center_jitter"], self.local_policy["center_jitter"]),
                0.0,
                1.0,
            )
            box = self._sample_random_box(
                image.size[0],
                image.size[1],
                scale_range=self.local_policy["anchor_dental_scale"],
                center=(center_x, center_y),
            )
            last_box = box
            if not self.local_policy["avoid_black_background"] or self._box_foreground_ratio(gray_map, box) >= self.local_policy["min_foreground_ratio"]:
                crop = image.crop(box).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
                return crop, box

        fallback_box = self._sample_box_with_foreground_constraint(
            image,
            gray_map,
            scale_range=self.local_policy["random_foreground_scale"],
            fallback_box=last_box,
        )
        crop = image.crop(fallback_box).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, fallback_box

    def _sample_box_with_foreground_constraint(self, image, gray_map, scale_range, fallback_box=None):
        max_attempts = max(self.local_policy["max_resample_attempts"], 1)
        last_box = fallback_box
        for _ in range(max_attempts):
            box = self._sample_random_box(image.size[0], image.size[1], scale_range=scale_range)
            last_box = box
            if not self.local_policy["avoid_black_background"] or self._box_foreground_ratio(gray_map, box) >= self.local_policy["min_foreground_ratio"]:
                return box
        if last_box is not None:
            return last_box
        return self._sample_random_box(image.size[0], image.size[1], scale_range=scale_range)

    def _sample_random_box(self, width, height, scale_range, ratio_range=(3.0 / 4.0, 4.0 / 3.0), center=None):
        area = float(width * height)
        log_ratio = (np.log(ratio_range[0]), np.log(ratio_range[1]))
        for _ in range(10):
            target_area = area * random.uniform(*scale_range)
            aspect_ratio = np.exp(random.uniform(*log_ratio))
            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
            if crop_w < width and crop_h < height and crop_w > 1 and crop_h > 1:
                return self._box_from_size(width, height, crop_w, crop_h, center=center)

        min_side = min(width, height)
        crop_size = int(round(np.clip(np.sqrt(area * sum(scale_range) * 0.5), 2, min_side)))
        return self._box_from_size(width, height, crop_size, crop_size, center=center)

    def _box_from_size(self, width, height, crop_w, crop_h, center=None):
        crop_w = min(max(int(crop_w), 2), width)
        crop_h = min(max(int(crop_h), 2), height)
        if center is None:
            x0 = 0 if width == crop_w else random.randint(0, width - crop_w)
            y0 = 0 if height == crop_h else random.randint(0, height - crop_h)
        else:
            center_x = float(center[0]) * width
            center_y = float(center[1]) * height
            x0 = int(np.clip(round(center_x - crop_w / 2), 0, max(width - crop_w, 0)))
            y0 = int(np.clip(round(center_y - crop_h / 2), 0, max(height - crop_h, 0)))
        return (x0, y0, x0 + crop_w, y0 + crop_h)

    def _get_grayscale_map(self, image):
        gray = ImageOps.grayscale(image)
        return np.asarray(gray, dtype=np.float32) / 255.0

    def _box_foreground_ratio(self, gray_map, box):
        x0, y0, x1, y1 = [int(v) for v in box]
        patch = gray_map[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float((patch > self.foreground_intensity_threshold).mean())

    def save_debug_local_crops_grid(self, image, output_path, columns=4):
        sample = self(image)
        local_crops = sample["local_crops"]
        pil_crops = [self._tensor_to_debug_pil(crop) for crop in local_crops]
        columns = max(1, columns)
        rows = int(np.ceil(len(pil_crops) / columns))
        tile_w, tile_h = pil_crops[0].size
        canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), color=(0, 0, 0))
        for idx, crop in enumerate(pil_crops):
            x = (idx % columns) * tile_w
            y = (idx // columns) * tile_h
            canvas.paste(crop, (x, y))
        canvas.save(output_path)
        return output_path

    def _tensor_to_debug_pil(self, tensor):
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
        image = tensor.detach().cpu() * std.cpu() + mean.cpu()
        image = image.clamp(0.0, 1.0)
        image = (image * 255.0).byte().permute(1, 2, 0).numpy()
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image)

    def _sample_quadrant_crop(self, image):
        width, height = image.size
        quadrant = random.randint(0, 3)
        half_w = width // 2
        half_h = height // 2
        x0 = 0 if quadrant % 2 == 0 else half_w
        y0 = 0 if quadrant < 2 else half_h
        x1 = half_w if quadrant % 2 == 0 else width
        y1 = half_h if quadrant < 2 else height
        return self._crop_with_jitter(image, x0, y0, x1, y1)

    def _sample_balanced_quadrant_crop(self, image, center):
        width, height = image.size
        center_x = width * center[0] + random.uniform(-0.06, 0.06) * width
        center_y = height * center[1] + random.uniform(-0.06, 0.06) * height
        crop_w = int(np.clip(width * random.uniform(0.20, 0.30), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.24, 0.36), self.local_crops_size, height))
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_balanced_arch_patch(self, image, center):
        width, height = image.size
        center_x = width * center[0] + random.uniform(-0.05, 0.05) * width
        center_y = height * center[1] + random.uniform(-0.05, 0.05) * height
        crop_w = int(np.clip(width * random.uniform(0.18, 0.26), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.20, 0.30), self.local_crops_size, height))
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_balanced_transition_crop(self, image, center):
        width, height = image.size
        center_x = width * center[0] + random.uniform(-0.05, 0.05) * width
        center_y = height * center[1] + random.uniform(-0.05, 0.05) * height
        crop_w = int(np.clip(width * random.uniform(0.18, 0.24), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.22, 0.30), self.local_crops_size, height))
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_compact_quadrant_crop(self, image):
        width, height = image.size
        quadrant = random.randint(0, 3)
        half_w = width / 2.0
        half_h = height / 2.0
        center_x = width * (0.25 if quadrant % 2 == 0 else 0.75)
        center_y = height * (0.25 if quadrant < 2 else 0.75)
        crop_w = int(np.clip(width * random.uniform(0.22, 0.34), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.28, 0.42), self.local_crops_size, height))
        center_x += random.uniform(-0.12, 0.12) * half_w
        center_y += random.uniform(-0.12, 0.12) * half_h
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_arch_patch(self, image):
        width, height = image.size
        target_size = int(self.local_crops_size / max(self.local_crop_min_scale, 1e-6))
        crop_size = int(
            np.clip(
                target_size * random.uniform(self.local_crop_min_scale, self.local_crop_max_scale),
                self.local_crops_size,
                min(width, height),
            )
        )
        center_x = width * (0.5 + random.uniform(-self.local_crop_jitter, self.local_crop_jitter))
        arch_center_y = 0.5 * height
        center_bias = (random.random() - 0.5) * 2.0 * self.local_crop_jitter * height
        center_y = arch_center_y + center_bias * self.local_arch_focus_strength

        x0 = int(np.clip(center_x - crop_size / 2, 0, max(width - crop_size, 0)))
        y0 = int(np.clip(center_y - crop_size / 2, 0, max(height - crop_size, 0)))
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        crop = image.crop((x0, y0, x1, y1)).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, (x0, y0, x1, y1)

    def _sample_compact_arch_patch(self, image):
        width, height = image.size
        crop_w = int(np.clip(width * random.uniform(0.20, 0.32), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.22, 0.34), self.local_crops_size, height))
        center_x = width * random.choice((0.22, 0.38, 0.50, 0.62, 0.78))
        center_x += random.uniform(-0.09, 0.09) * width
        center_y = height * random.choice((0.42, 0.52, 0.62))
        center_y += random.uniform(-0.06, 0.06) * height
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_transition_crop(self, image):
        width, height = image.size
        target_size = int(self.local_crops_size / max(self.local_crop_min_scale, 1e-6))
        crop_size = int(
            np.clip(
                target_size * random.uniform(max(self.local_crop_min_scale, 0.75), self.local_crop_max_scale),
                self.local_crops_size,
                min(width, height),
            )
        )
        center_x = width * random.choice((0.35, 0.5, 0.65))
        center_x += random.uniform(-self.local_crop_jitter, self.local_crop_jitter) * width * 0.5
        center_y = height * (0.5 + random.choice((-0.14, 0.14)))
        center_y += random.uniform(-self.local_crop_jitter, self.local_crop_jitter) * height * 0.35
        x0 = int(np.clip(center_x - crop_size / 2, 0, max(width - crop_size, 0)))
        y0 = int(np.clip(center_y - crop_size / 2, 0, max(height - crop_size, 0)))
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        crop = image.crop((x0, y0, x1, y1)).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, (x0, y0, x1, y1)

    def _sample_compact_transition_crop(self, image):
        width, height = image.size
        crop_w = int(np.clip(width * random.uniform(0.18, 0.28), self.local_crops_size, width))
        crop_h = int(np.clip(height * random.uniform(0.24, 0.34), self.local_crops_size, height))
        center_x = width * random.choice((0.34, 0.50, 0.66))
        center_x += random.uniform(-0.08, 0.08) * width
        center_y = height * random.choice((0.40, 0.50, 0.60))
        center_y += random.uniform(-0.06, 0.06) * height
        return self._crop_from_center(image, center_x, center_y, crop_w, crop_h)

    def _sample_toothband_guided_crop(
        self,
        image,
        *,
        center,
        width_range,
        height_range,
        existing_boxes=None,
        iou_threshold=0.15,
    ):
        gray, saliency, (_, _, focus_y) = self._build_tooth_focus_maps(image)
        width, height = image.size
        min_crop = max(self.local_crops_size // 2, 32)
        best = None
        best_score = None
        for _ in range(16):
            center_x = width * center[0] + random.uniform(-0.04, 0.04) * width
            center_y = height * center[1] + random.uniform(-0.04, 0.04) * height
            crop_w = int(np.clip(width * random.uniform(*width_range), min_crop, width))
            crop_h = int(np.clip(height * random.uniform(*height_range), min_crop, height))
            crop, box = self._crop_from_center(image, center_x, center_y, crop_w, crop_h)
            overlaps = [self._box_iou(box, prev_box) for prev_box in (existing_boxes or [])]
            max_overlap = max(overlaps) if overlaps else 0.0
            overlap_penalty = sum(overlaps)
            score = self._score_crop_structure(image, box, gray=gray, saliency=saliency, focus_y=focus_y)
            score -= 0.55 * overlap_penalty
            if max_overlap > max(iou_threshold, 0.22):
                score -= 0.40
            if best is None or score > best_score:
                best = (crop, box)
                best_score = score
        return best

    def _sample_subcrop_within_parent(
        self,
        image,
        parent_box,
        *,
        existing_boxes=None,
        scale_range=(0.34, 0.48),
        iou_threshold=0.06,
    ):
        gray, saliency, (_, _, focus_y) = self._build_tooth_focus_maps(image)
        px0, py0, px1, py1 = [int(round(v)) for v in parent_box]
        parent_w = max(px1 - px0, 2)
        parent_h = max(py1 - py0, 2)
        min_side = max(self.local_crops_size // 3, int(min(parent_w, parent_h) * 0.30))
        best = None
        best_score = None
        for _ in range(16):
            crop_w = int(np.clip(parent_w * random.uniform(*scale_range), min_side, parent_w))
            crop_h = int(np.clip(parent_h * random.uniform(*scale_range), min_side, parent_h))
            if parent_w == crop_w:
                x0 = px0
            else:
                x0 = random.randint(px0, px1 - crop_w)
            if parent_h == crop_h:
                y0 = py0
            else:
                y0 = random.randint(py0, py1 - crop_h)
            box = (x0, y0, x0 + crop_w, y0 + crop_h)
            crop = image.crop(box).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
            overlaps = [self._box_iou(box, prev_box) for prev_box in (existing_boxes or [])]
            overlap_penalty = sum(overlaps)
            max_overlap = max(overlaps) if overlaps else 0.0
            score = self._score_crop_structure(image, box, gray=gray, saliency=saliency, focus_y=focus_y)
            score -= 0.60 * overlap_penalty
            if max_overlap > max(iou_threshold, 0.20):
                score -= 0.40
            if best is None or score > best_score:
                best = (crop, box)
                best_score = score
        return best

    def _sample_anatomy_patch(self, image):
        width, height = image.size
        target_size = int(self.local_crops_size / max(self.local_crop_min_scale, 1e-6))
        crop_size = int(
            np.clip(
                target_size * random.uniform(self.local_crop_min_scale, self.local_crop_max_scale),
                self.local_crops_size,
                min(width, height),
            )
        )

        heatmap = self._build_mask_bias_map(image, max(width, height) // self.patch_size)
        heatmap_np = heatmap.cpu().numpy()
        flat_idx = np.random.choice(heatmap_np.size, p=heatmap_np.reshape(-1))
        grid_h, grid_w = heatmap_np.shape
        grid_y, grid_x = np.unravel_index(flat_idx, (grid_h, grid_w))
        center_x = ((grid_x + 0.5) / grid_w) * width
        center_y = ((grid_y + 0.5) / grid_h) * height

        x0 = int(np.clip(center_x - crop_size / 2, 0, max(width - crop_size, 0)))
        y0 = int(np.clip(center_y - crop_size / 2, 0, max(height - crop_size, 0)))
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        crop = image.crop((x0, y0, x1, y1)).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, (x0, y0, x1, y1)

    def _crop_from_center(self, image, center_x, center_y, crop_w, crop_h):
        width, height = image.size
        crop_w = min(max(int(crop_w), 2), width)
        crop_h = min(max(int(crop_h), 2), height)
        x0 = int(np.clip(round(center_x - crop_w / 2), 0, max(width - crop_w, 0)))
        y0 = int(np.clip(round(center_y - crop_h / 2), 0, max(height - crop_h, 0)))
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        crop = image.crop((x0, y0, x1, y1)).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, (x0, y0, x1, y1)

    def _crop_with_jitter(self, image, x0, y0, x1, y1):
        width, height = image.size
        box_w = x1 - x0
        box_h = y1 - y0
        jitter_x = int(box_w * self.local_crop_jitter)
        jitter_y = int(box_h * self.local_crop_jitter)
        x0 = max(0, x0 + random.randint(-jitter_x, jitter_x))
        y0 = max(0, y0 + random.randint(-jitter_y, jitter_y))
        x1 = min(width, x1 + random.randint(-jitter_x, jitter_x))
        y1 = min(height, y1 + random.randint(-jitter_y, jitter_y))
        if x1 <= x0 + 1:
            x1 = min(width, x0 + 2)
        if y1 <= y0 + 1:
            y1 = min(height, y0 + 2)
        crop = image.crop((x0, y0, x1, y1)).resize((self.local_crops_size, self.local_crops_size), resample=Image.BICUBIC)
        return crop, (x0, y0, x1, y1)

    def _encode_side_label(self, width, box):
        x0, _, x1, _ = box
        center_x = 0.5 * (x0 + x1) / max(width, 1)
        if center_x < 1.0 / 3.0:
            return 0
        if center_x < 2.0 / 3.0:
            return 1
        return 2
