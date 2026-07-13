# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random

import torch


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "global_flip_labels" in samples_list[0][0]:
        collated_global_flip_labels = torch.tensor(
            [s[0]["global_flip_labels"][i] for i in range(n_global_crops) for s in samples_list],
            dtype=torch.float32,
        )
    else:
        collated_global_flip_labels = None
    if "local_side_labels" in samples_list[0][0]:
        collated_local_side_labels = torch.tensor(
            [s[0]["local_side_labels"][i] for i in range(n_local_crops) for s in samples_list],
            dtype=torch.long,
        )
    else:
        collated_local_side_labels = None
    if "local_region_labels" in samples_list[0][0]:
        collated_local_region_labels = torch.tensor(
            [s[0]["local_region_labels"][i] for i in range(n_local_crops) for s in samples_list],
            dtype=torch.long,
        )
    else:
        collated_local_region_labels = None
    if "local_crop_boxes" in samples_list[0][0]:
        collated_local_crop_boxes = torch.tensor(
            [s[0]["local_crop_boxes"][i] for i in range(n_local_crops) for s in samples_list],
            dtype=torch.float32,
        )
    else:
        collated_local_crop_boxes = None
    if "local_source_global_indices" in samples_list[0][0]:
        collated_local_source_global_indices = torch.tensor(
            [s[0]["local_source_global_indices"][i] for i in range(n_local_crops) for s in samples_list],
            dtype=torch.long,
        )
    else:
        collated_local_source_global_indices = None
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None
    if "mask_bias_maps" in samples_list[0][0]:
        collated_mask_bias_maps = torch.stack(
            [s[0]["mask_bias_maps"][i] for i in range(n_global_crops) for s in samples_list]
        )
    else:
        collated_mask_bias_maps = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = [None for _ in range(B)]
    if collated_mask_bias_maps is None:
        masked_indices = list(range(n_samples_masked))
    else:
        masked_indices = random.sample(range(B), n_samples_masked)
    for i, sample_index in enumerate(masked_indices):
        prob_max = probs[i + 1]
        bias_map = None if collated_mask_bias_maps is None else collated_mask_bias_maps[sample_index].cpu().numpy()
        mask = torch.BoolTensor(mask_generator(int(N * prob_max), bias_map=bias_map))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list[sample_index] = mask
        upperbound += int(N * prob_max)
    for sample_index in range(B):
        if masks_list[sample_index] is None:
            masks_list[sample_index] = torch.BoolTensor(mask_generator(0))

    if collated_mask_bias_maps is None:
        random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    if collated_global_flip_labels is not None:
        out["collated_global_flip_labels"] = collated_global_flip_labels
    if collated_local_side_labels is not None:
        out["collated_local_side_labels"] = collated_local_side_labels
    if collated_local_region_labels is not None:
        out["collated_local_region_labels"] = collated_local_region_labels
    if collated_local_crop_boxes is not None:
        out["collated_local_crop_boxes"] = collated_local_crop_boxes
    if collated_local_source_global_indices is not None:
        out["collated_local_source_global_indices"] = collated_local_source_global_indices
    if collated_mask_bias_maps is not None:
        out["collated_mask_bias_maps"] = collated_mask_bias_maps
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if "collated_global_flip_labels" in collated_data_batch:
        flip_old_bs = collated_data_batch["collated_global_flip_labels"].shape[0] // 2
        new_batch["collated_global_flip_labels"] = (
            collated_data_batch["collated_global_flip_labels"]
            .unflatten(0, (2, flip_old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
    if "collated_local_side_labels" in collated_data_batch:
        local_old_bs = collated_data_batch["collated_local_side_labels"].shape[0] // (
            collated_data_batch["collated_local_crops"].shape[0] // old_bs
        )
        n_local_crops = collated_data_batch["collated_local_crops"].shape[0] // old_bs
        new_batch["collated_local_side_labels"] = (
            collated_data_batch["collated_local_side_labels"]
            .unflatten(0, (n_local_crops, local_old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
    if "collated_local_region_labels" in collated_data_batch:
        local_old_bs = collated_data_batch["collated_local_region_labels"].shape[0] // (
            collated_data_batch["collated_local_crops"].shape[0] // old_bs
        )
        n_local_crops = collated_data_batch["collated_local_crops"].shape[0] // old_bs
        new_batch["collated_local_region_labels"] = (
            collated_data_batch["collated_local_region_labels"]
            .unflatten(0, (n_local_crops, local_old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
    if "collated_local_crop_boxes" in collated_data_batch:
        n_local_crops = collated_data_batch["collated_local_crops"].shape[0] // old_bs
        new_batch["collated_local_crop_boxes"] = (
            collated_data_batch["collated_local_crop_boxes"]
            .unflatten(0, (n_local_crops, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
    if "collated_local_source_global_indices" in collated_data_batch:
        n_local_crops = collated_data_batch["collated_local_crops"].shape[0] // old_bs
        new_batch["collated_local_source_global_indices"] = (
            collated_data_batch["collated_local_source_global_indices"]
            .unflatten(0, (n_local_crops, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by
    if "collated_mask_bias_maps" in collated_data_batch:
        bias_old_bs = collated_data_batch["collated_mask_bias_maps"].shape[0] // 2
        new_batch["collated_mask_bias_maps"] = (
            collated_data_batch["collated_mask_bias_maps"]
            .unflatten(0, (2, bias_old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )

    return new_batch
