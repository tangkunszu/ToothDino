# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
from functools import partial
from pathlib import Path

import torch
import torch.distributed as torch_dist
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor

import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint
from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, KoLeoLossDistributed, iBOTPatchLoss
from dinov3.models import build_model_from_cfg
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters

logger = logging.getLogger("dinov3")

def _patchify_for_stage_a(images: Tensor, patch_size: int) -> Tensor:
    # TRACE_STAGEA_S3_MIM: convert images [B, C, H, W] into patch targets [B, P, patch_area * C].
    B, C, H, W = images.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"TRACE_STAGEA_S3_MIM: image size {(H, W)} must be divisible by patch size {patch_size}"
        )
    h = H // patch_size
    w = W // patch_size
    x = images.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, patch_size * patch_size * C)


def _tokens_to_patch_images(tokens: Tensor, patch_size: int, channels: int) -> Tensor:
    expected_dim = patch_size * patch_size * channels
    if tokens.shape[-1] != expected_dim:
        raise ValueError(
            f"TRACE_STAGEA_S3_MIM: token dim {tokens.shape[-1]} must equal patch dim {expected_dim}"
        )
    return tokens.reshape(tokens.shape[0], patch_size, patch_size, channels).permute(0, 3, 1, 2).contiguous()


def _to_grayscale(patches: Tensor) -> Tensor:
    if patches.shape[1] == 1:
        return patches
    return patches.mean(dim=1, keepdim=True)


def _sobel_gradient_magnitude(patches: Tensor) -> Tensor:
    gray = _to_grayscale(patches.float())
    kernel_x = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device, dtype=gray.dtype)
        .view(1, 1, 3, 3)
        .div_(4.0)
    )
    kernel_y = (
        torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device, dtype=gray.dtype)
        .view(1, 1, 3, 3)
        .div_(4.0)
    )
    grad_x = torch.nn.functional.conv2d(gray, kernel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1.0e-6)


def _build_transition_weights(target_patches: Tensor, transition_weight: float, transition_gamma: float) -> tuple[Tensor, Tensor]:
    edge_energy = _sobel_gradient_magnitude(target_patches).mean(dim=(1, 2, 3))
    if edge_energy.numel() <= 1 or transition_weight <= 0:
        normalized = torch.zeros_like(edge_energy)
    else:
        normalized = edge_energy - edge_energy.amin()
        normalized = normalized / normalized.amax().clamp_min(1.0e-6)
    weights = 1.0 + transition_weight * normalized.pow(transition_gamma)
    return weights, edge_energy


def _frequency_filter_images(
    images: Tensor,
    *,
    mode: str,
    cutoff: float,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> Tensor:
    input_dtype = images.dtype
    images_f = images.float()
    mean_t = torch.as_tensor(mean, device=images.device, dtype=images_f.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=images.device, dtype=images_f.dtype).view(1, -1, 1, 1)
    raw = (images_f * std_t + mean_t).clamp(0.0, 1.0)

    _, _, H, W = raw.shape
    ys = torch.arange(H, device=images.device, dtype=images_f.dtype) - H // 2
    xs = torch.arange(W, device=images.device, dtype=images_f.dtype) - W // 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    radius = torch.sqrt(yy.pow(2) + xx.pow(2))
    radius = radius / (torch.sqrt(torch.as_tensor((H / 2) ** 2 + (W / 2) ** 2, device=images.device, dtype=images_f.dtype)) + 1.0e-6)

    if mode == "low":
        mask = radius <= cutoff
    elif mode == "high":
        mask = radius >= cutoff
    else:
        raise ValueError(f"Unsupported frequency filter mode={mode}")

    fft = torch.fft.fftshift(torch.fft.fft2(raw, dim=(-2, -1)), dim=(-2, -1))
    filtered_fft = fft * mask.view(1, 1, H, W)
    filtered = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft, dim=(-2, -1)), dim=(-2, -1)).real
    if mode == "high":
        filtered = filtered + 0.5
    filtered = filtered.clamp(0.0, 1.0)
    normalized = (filtered - mean_t) / std_t
    return normalized.to(input_dtype)


def _normalize_anatomy_prior(prior: Tensor, mode: str, temperature: float) -> Tensor:
    if mode == "softmax":
        return torch.softmax(prior / max(temperature, 1e-6), dim=-1)
    if mode == "minmax":
        prior = prior - prior.amin(dim=-1, keepdim=True)
        prior = prior / prior.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return prior
    raise ValueError(f"Unsupported anatomy_discovery.normalize={mode}")


def _mix_patch_priors(primary: Tensor | None, secondary: Tensor | None, mix: float) -> Tensor | None:
    if primary is None:
        return secondary
    if secondary is None or mix <= 0:
        return primary
    mix = float(min(max(mix, 0.0), 1.0))
    return (1.0 - mix) * primary + mix * secondary


class SSLMetaArch(nn.Module):
    """
    Modified version of SSLMetaArchCompilable including gram loss:
    - Gram loss is used only if gram.use_loss is set to true
    """

    def __init__(self, cfg):
        super().__init__()

        # assert cfg.multidistillation.enabled is False
        assert cfg.crops.local_crops_number > 0
        assert cfg.ibot.separate_head is True
        assert cfg.train.centering == "sinkhorn_knopp"

        # For some reason FULL_SHARD doesn't work
        assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        gram_model_dict["backbone"] = gram_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {cfg.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {cfg.dino.koleo_loss_distributed}")
        if cfg.dino.koleo_loss_distributed:
            logger.info(f"OPTIONS -- KOLEO -- topk: {cfg.dino.koleo_topk}")
            logger.info(
                f"OPTIONS -- KOLEO -- distributed_loss_group_size: {cfg.dino.koleo_distributed_loss_group_size}"
            )
            assert cfg.dino.koleo_distributed_replicas == 0, (
                "Option `dino.koleo_distributed_replicas` is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=cfg.dino.koleo_topk,
                loss_group_size=cfg.dino.koleo_distributed_loss_group_size,
            )
        else:
            assert cfg.dino.koleo_topk == 1, "Non-distributed KoLeo loss only supports `dino.koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        assert 0 <= cfg.ibot.mask_ratio_min_max[0] < cfg.ibot.mask_ratio_min_max[1] <= 1, (
            "provide a valid cfg.ibot.mask_ratio_min_max"
        )
        assert 0 <= cfg.ibot.mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
        logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_norm_last_layer: {cfg.ibot.head_norm_last_layer}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)
        self.anatomy_discovery_enabled = cfg.anatomy_discovery.enabled
        self.anatomy_discovery_source = cfg.anatomy_discovery.source
        self.anatomy_discovery_normalize = cfg.anatomy_discovery.normalize
        self.anatomy_discovery_temperature = cfg.anatomy_discovery.temperature
        self.anatomy_discovery_mix_with_static_prior = cfg.anatomy_discovery.mix_with_static_prior
        self.last_token_prior_cfg = getattr(cfg, "last_token_prior", None)
        self.last_token_prior_enabled = bool(getattr(self.last_token_prior_cfg, "enabled", False))
        self.last_token_prior_normalize = getattr(self.last_token_prior_cfg, "normalize", "softmax")
        self.last_token_prior_temperature = getattr(self.last_token_prior_cfg, "temperature", 1.0)
        self.last_token_prior_cls_mix = float(getattr(self.last_token_prior_cfg, "cls_mix", 0.0))
        self.last_token_prior_smooth_kernel = int(getattr(self.last_token_prior_cfg, "smooth_kernel", 3))
        self.last_token_prior_use_for_anatomy_discovery = bool(
            getattr(self.last_token_prior_cfg, "use_for_anatomy_discovery", False)
        )
        self.last_token_prior_use_for_token_correspondence = bool(
            getattr(self.last_token_prior_cfg, "use_for_token_correspondence", False)
        )
        self.last_token_prior_use_for_masking = bool(getattr(self.last_token_prior_cfg, "use_for_masking", False))
        if self.last_token_prior_enabled:
            logger.info("OPTIONS -- LAST TOKEN PRIOR")
            logger.info("OPTIONS -- LAST TOKEN PRIOR -- cls_mix: %s", self.last_token_prior_cls_mix)
            logger.info("OPTIONS -- LAST TOKEN PRIOR -- temperature: %s", self.last_token_prior_temperature)
            logger.info("OPTIONS -- LAST TOKEN PRIOR -- smooth_kernel: %s", self.last_token_prior_smooth_kernel)
            logger.info(
                "OPTIONS -- LAST TOKEN PRIOR -- use_for_anatomy_discovery: %s",
                self.last_token_prior_use_for_anatomy_discovery,
            )
            logger.info(
                "OPTIONS -- LAST TOKEN PRIOR -- use_for_token_correspondence: %s",
                self.last_token_prior_use_for_token_correspondence,
            )
            if self.last_token_prior_use_for_masking:
                logger.warning(
                    "LAST TOKEN PRIOR for masking is configured, but masking happens before forward; "
                    "this implementation does not yet apply LAST prior to data-side masking."
                )
        if self.anatomy_discovery_enabled:
            logger.info("OPTIONS -- ANATOMY DISCOVERY")
            logger.info(f"OPTIONS -- ANATOMY DISCOVERY -- source: {self.anatomy_discovery_source}")
            logger.info(f"OPTIONS -- ANATOMY DISCOVERY -- normalize: {self.anatomy_discovery_normalize}")
            logger.info(f"OPTIONS -- ANATOMY DISCOVERY -- temperature: {self.anatomy_discovery_temperature}")
            logger.info(
                "OPTIONS -- ANATOMY DISCOVERY -- mix_with_static_prior: %s",
                self.anatomy_discovery_mix_with_static_prior,
            )
        self.token_correspondence_enabled = cfg.token_correspondence.enabled
        self.token_correspondence_loss_weight = cfg.token_correspondence.loss_weight
        self.token_correspondence_loss_type = cfg.token_correspondence.loss_type
        self.token_correspondence_use_teacher_prior = cfg.token_correspondence.use_teacher_prior
        self.token_correspondence_detach_teacher = cfg.token_correspondence.detach_teacher
        self.token_correspondence_mirror_aware = (
            cfg.token_correspondence.mirror_aware if "mirror_aware" in cfg.token_correspondence else False
        )
        self.token_correspondence_cross_view_only = (
            cfg.token_correspondence.cross_view_only if "cross_view_only" in cfg.token_correspondence else True
        )
        self.token_correspondence_use_flip_labels = (
            cfg.token_correspondence.use_flip_labels if "use_flip_labels" in cfg.token_correspondence else True
        )
        if self.token_correspondence_enabled:
            logger.info("OPTIONS -- TOKEN CORRESPONDENCE")
            logger.info(f"OPTIONS -- TOKEN CORRESPONDENCE -- loss_weight: {self.token_correspondence_loss_weight}")
            logger.info(f"OPTIONS -- TOKEN CORRESPONDENCE -- loss_type: {self.token_correspondence_loss_type}")
            logger.info(
                f"OPTIONS -- TOKEN CORRESPONDENCE -- use_teacher_prior: {self.token_correspondence_use_teacher_prior}"
            )
            logger.info(
                f"OPTIONS -- TOKEN CORRESPONDENCE -- detach_teacher: {self.token_correspondence_detach_teacher}"
            )
            logger.info(
                "OPTIONS -- TOKEN CORRESPONDENCE -- mirror_aware: %s",
                self.token_correspondence_mirror_aware,
            )
            logger.info(
                "OPTIONS -- TOKEN CORRESPONDENCE -- cross_view_only: %s",
                self.token_correspondence_cross_view_only,
            )
            logger.info(
                "OPTIONS -- TOKEN CORRESPONDENCE -- use_flip_labels: %s",
                self.token_correspondence_use_flip_labels,
            )
        self.flip_prediction_enabled = cfg.auxiliary.flip_prediction.enabled
        self.flip_prediction_loss_weight = cfg.auxiliary.flip_prediction.loss_weight
        self.flip_prediction_loss = nn.BCEWithLogitsLoss()
        if self.flip_prediction_enabled:
            student_model_dict["flip_head"] = nn.Linear(embed_dim, 1)
            teacher_model_dict["flip_head"] = nn.Linear(embed_dim, 1)
        local_side_cfg = getattr(cfg.auxiliary, "local_side_prediction", None)
        self.local_side_prediction_enabled = bool(local_side_cfg.enabled) if local_side_cfg is not None else False
        self.local_side_prediction_loss_weight = (
            local_side_cfg.loss_weight if local_side_cfg is not None else 0.1
        )
        self.local_side_prediction_num_classes = (
            local_side_cfg.num_classes if local_side_cfg is not None else 3
        )
        self.local_side_prediction_loss = nn.CrossEntropyLoss()
        if self.local_side_prediction_enabled:
            student_model_dict["local_side_head"] = nn.Linear(embed_dim, self.local_side_prediction_num_classes)
            teacher_model_dict["local_side_head"] = nn.Linear(embed_dim, self.local_side_prediction_num_classes)

        # --- Flip CLS Symmetry Loss ---
        # Dense embedding-space cross-crop alignment with flip-state awareness.
        # Rationale: The DINO global loss aligns views in prototype space (K=65536 dims).
        # This loss adds a complementary alignment directly in the representation space
        # (D-dim, typically 768 or 1024), conditioned on whether the two global crops
        # were flipped differently, which corresponds to anatomically mirrored views of
        # the same dental panoramic. Pairs with opposite flip states receive a higher
        # weight since they are true left-right symmetric views.
        flip_cls_sym_cfg = getattr(cfg.auxiliary, "flip_cls_symmetry", None)
        self.flip_cls_symmetry_enabled = bool(flip_cls_sym_cfg.enabled) if flip_cls_sym_cfg is not None else False
        self.flip_cls_symmetry_loss_weight = (
            flip_cls_sym_cfg.loss_weight if flip_cls_sym_cfg is not None else 0.1
        )
        self.flip_cls_symmetry_flip_bonus = (
            flip_cls_sym_cfg.flip_bonus if flip_cls_sym_cfg is not None else 0.5
        )
        if self.flip_cls_symmetry_enabled:
            logger.info("OPTIONS -- FLIP CLS SYMMETRY")
            logger.info(f"OPTIONS -- FLIP CLS SYMMETRY -- loss_weight: {self.flip_cls_symmetry_loss_weight}")
            logger.info(f"OPTIONS -- FLIP CLS SYMMETRY -- flip_bonus: {self.flip_cls_symmetry_flip_bonus}")

        self.mim_enabled = cfg.mim.enabled
        self.mim_loss_weight = cfg.mim.loss_weight
        self.mim_normalize_targets = cfg.mim.normalize_targets
        self.mim_loss_type = cfg.mim.loss_type
        self.mim_variant = cfg.mim.variant
        self.mim_edge_loss_weight = cfg.mim.edge_loss_weight
        self.mim_transition_weight = cfg.mim.transition_weight
        self.mim_transition_gamma = cfg.mim.transition_gamma
        if self.mim_enabled:
            # TRACE_STAGEA_S3_MIM: lightweight masked patch reconstruction head for dental panoramic pretraining.
            patch_dim = cfg.student.patch_size * cfg.student.patch_size * cfg.student.in_chans
            student_model_dict["mim_head"] = nn.Linear(embed_dim, patch_dim)
            teacher_model_dict["mim_head"] = nn.Linear(embed_dim, patch_dim)
            if self.mim_loss_type != "mse":
                raise ValueError(f"TRACE_STAGEA_S3_MIM: unsupported mim.loss_type={self.mim_loss_type}")
            if self.mim_variant not in {"pixel", "boundary_transition_aware"}:
                raise ValueError(f"TRACE_STAGEA_S3_MIM: unsupported mim.variant={self.mim_variant}")
            logger.info("OPTIONS -- MIM")
            logger.info(f"OPTIONS -- MIM -- enabled: {self.mim_enabled}")
            logger.info(f"OPTIONS -- MIM -- loss_weight: {self.mim_loss_weight}")
            logger.info(f"OPTIONS -- MIM -- normalize_targets: {self.mim_normalize_targets}")
            logger.info(f"OPTIONS -- MIM -- variant: {self.mim_variant}")
            logger.info(f"OPTIONS -- MIM -- edge_loss_weight: {self.mim_edge_loss_weight}")
            logger.info(f"OPTIONS -- MIM -- transition_weight: {self.mim_transition_weight}")
            logger.info(f"OPTIONS -- MIM -- transition_gamma: {self.mim_transition_gamma}")
            logger.info(f"OPTIONS -- MIM -- patch_dim: {patch_dim}")

        self.bandpass_consistency_enabled = cfg.bandpass_consistency.enabled
        self.bandpass_consistency_loss_weight = cfg.bandpass_consistency.loss_weight
        self.bandpass_low_cutoff = cfg.bandpass_consistency.low_cutoff
        self.bandpass_high_cutoff = cfg.bandpass_consistency.high_cutoff
        self.bandpass_cls_loss_weight = cfg.bandpass_consistency.cls_loss_weight
        self.bandpass_patch_loss_weight = cfg.bandpass_consistency.patch_loss_weight
        if self.bandpass_consistency_enabled:
            logger.info("OPTIONS -- BANDPASS CONSISTENCY")
            logger.info(f"OPTIONS -- BANDPASS CONSISTENCY -- loss_weight: {self.bandpass_consistency_loss_weight}")
            logger.info(f"OPTIONS -- BANDPASS CONSISTENCY -- low_cutoff: {self.bandpass_low_cutoff}")
            logger.info(f"OPTIONS -- BANDPASS CONSISTENCY -- high_cutoff: {self.bandpass_high_cutoff}")
            logger.info(f"OPTIONS -- BANDPASS CONSISTENCY -- cls_loss_weight: {self.bandpass_cls_loss_weight}")
            logger.info(f"OPTIONS -- BANDPASS CONSISTENCY -- patch_loss_weight: {self.bandpass_patch_loss_weight}")

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # getting config params fixed:
        self.n_local_crops = self.cfg.crops.local_crops_number
        self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )

        # Gram
        self.gram_use_loss = self.cfg.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            # Gram regularization
            self.gram_loss = GramLoss(
                apply_norm=self.cfg.gram.normalized,
                remove_only_teacher_neg=self.cfg.gram.remove_only_teacher_neg,
                remove_neg=self.cfg.gram.remove_neg,
            )
            # Construct gram teacher
            self.has_gram_teacher = True if not cfg.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")
            else:
                self.gram_teacher = None

            self.gram_loss_weight = self.cfg.gram.loss_weight
            if self.cfg.gram.get("loss_weight_schedule"):
                iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * cfg.optim.epochs
                schedule_cfg = self.cfg.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_cfg.start,
                    peak=schedule_cfg.peak,
                    end=schedule_cfg.end,
                    warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                    ),
                )
                logger.info(f"Applying gram loss weight schedule instead of `cfg.gram.loss_weight`: {schedule_cfg}")
            else:
                self.gram_loss_schedule = None
            self.gram_ema_teacher = self.cfg.gram.ema_teacher  # If true use the EMA_teacher as gram_teacher
            self.gram_ckpt = self.cfg.gram.ckpt  # Checkpoint to the first gram teacher model
            self.gram_img_level = self.cfg.gram.img_level  # Apply the loss on the image, if false on the batch
            self.gram_tokens_used = self.cfg.gram.tokens_used  # Any value in ["all", "masked", "unmasked"]
            # Update the teacher frequently
            self.gram_rep_update = self.cfg.gram.rep_update  # bool, if yes the gram teacher will be updated at the freq
            self.gram_update_frequency = self.cfg.gram.update_frequency  # defined by this var update_frequency
            self.gram_it_first_update = self.cfg.gram.it_first_update  # after iteration it_first_update is passed.
            self.gram_it_load_ema_teacher = (
                self.cfg.gram.it_load_ema_teacher
            )  # after iteration it_load_ema the ema teacher is loaded into the gram teacher
            self.gram_compute_stats = self.cfg.gram.compute_stats  # whether to compute auxiliary stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {cfg.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {cfg.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {cfg.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {cfg.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}")

            if cfg.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size must be set to use gram loss")
            if cfg.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size shoud be None when gram.ema_teacher=True")

            self.student_crop_size = cfg.crops.global_crops_size
            self.gram_global_teacher_resize_method = cfg.gram.global_teacher_resize_method
            self.gram_global_teacher_resize_antialias = cfg.gram.global_teacher_resize_antialias
            logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher size: {cfg.crops.gram_teacher_crops_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {cfg.gram.global_teacher_resize_method}")
            logger.info(
                f"OPTIONS -- global crops GRAM teacher resize antialias: {cfg.gram.global_teacher_resize_antialias}"
            )

    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes
        assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes
        assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        if self.flip_prediction_enabled:
            nn.init.trunc_normal_(self.student.flip_head.weight, std=0.02)
            nn.init.constant_(self.student.flip_head.bias, 0)
            nn.init.trunc_normal_(self.teacher.flip_head.weight, std=0.02)
            nn.init.constant_(self.teacher.flip_head.bias, 0)
        if self.local_side_prediction_enabled:
            nn.init.trunc_normal_(self.student.local_side_head.weight, std=0.02)
            nn.init.constant_(self.student.local_side_head.bias, 0)
            nn.init.trunc_normal_(self.teacher.local_side_head.weight, std=0.02)
            nn.init.constant_(self.teacher.local_side_head.bias, 0)
        if self.mim_enabled:
            # TRACE_STAGEA_S3_MIM: init lightweight reconstruction heads.
            nn.init.trunc_normal_(self.student.mim_head.weight, std=0.02)
            nn.init.constant_(self.student.mim_head.bias, 0)
            nn.init.trunc_normal_(self.teacher.mim_head.weight, std=0.02)
            nn.init.constant_(self.teacher.mim_head.bias, 0)
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                init_fsdp_model_from_checkpoint(
                    self.gram_teacher,
                    self.gram_ckpt,
                    skip_load_keys=[
                        "dino_head",
                        "ibot_head",
                        "dino_loss.center",
                        "ibot_patch_loss.center",
                    ],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
                self.gram_teacher_initialized = True
            else:
                # TRACE_STAGEA_S5_LOSS_TUNING: allow GRAM teacher bootstrapping from the current EMA teacher
                # when no dedicated GRAM checkpoint is provided.
                logger.info("Initializing Gram teacher from current EMA teacher because gram.ckpt is not set.")
                self.gram_load_ema_teacher()
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        # Prefer explicit resume checkpoint; otherwise fall back to student.pretrained_weights
        resume_ckpt = self.cfg.student.resume_from_teacher_chkpt
        pretrained_ckpt = self.cfg.student.pretrained_weights
        ckpt_path_str = resume_ckpt or pretrained_ckpt
        if resume_ckpt and pretrained_ckpt:
            logger.warning(
                "Both student.resume_from_teacher_chkpt and student.pretrained_weights are set; "
                "using resume_from_teacher_chkpt=%s", resume_ckpt
            )
        if ckpt_path_str:
            source = (
                "student.resume_from_teacher_chkpt"
                if resume_ckpt
                else "student.pretrained_weights"
            )
            logger.info(f"Loading student weights from {ckpt_path_str} (source={source})")
            ckpt_path = Path(ckpt_path_str)
            if ckpt_path.is_dir():
                init_fsdp_model_from_checkpoint(
                    self.student,
                    str(ckpt_path),
                    skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                    process_group=distributed.get_process_subgroup(),
                )
                # No granular missing/unexpected info from directory checkpoint loader.
            else:
                # Fallback for backbone-only consolidated checkpoints: load into student with loose strictness
                try:
                    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                except TypeError:
                    # Older torch versions may not support weights_only
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    logger.warning("torch.load without weights_only=True; consider upgrading torch for safer loading.")
                state_dict = state_dict.get("teacher", state_dict)
                if all(not k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}
                # Convert tensors to DTensor if the model is sharded (avoid tensor/DTensor mismatch)
                if torch_dist.is_initialized():
                    process_group = distributed.get_process_subgroup()
                    world_mesh = DeviceMesh.from_group(process_group, "cuda")
                    keys_not_sharded = ["backbone.rope_embed.periods", "qkv.bias_mask"]

                    def maybe_shard(key, tensor):
                        if any(skip in key for skip in keys_not_sharded):
                            return tensor
                        return distribute_tensor(tensor, world_mesh)

                    state_dict = {k: maybe_shard(k, v) for k, v in state_dict.items()}
                msg = self.student.load_state_dict(state_dict, strict=False)
                loaded_keys = sorted(list(state_dict.keys()))[:5]
                missing, unexpected = msg.missing_keys, msg.unexpected_keys
                # Compute a scalar fingerprint even when parameters are DTensors.
                wt_sum = 0.0
                for p in self.student.parameters():
                    try:
                        wt_sum += p.detach().abs().mean().item()
                    except Exception:
                        # Fallback: convert to dense if needed
                        wt_sum += p.detach().to_local().abs().mean().item() if hasattr(p, "to_local") else 0.0
                logger.info(
                    "Student checkpoint loaded. strict=False msg=%s, total_keys=%d, sample_keys=%s",
                    msg,
                    len(state_dict),
                    loaded_keys,
                )
                logger.info(f"[DINOv3] Loaded student: missing_keys={missing}, unexpected_keys={unexpected}")
                logger.info(f"[DINOv3] post-load weight abs-mean fingerprint: {wt_sum:.3f}")
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_keys=["dino_loss.center", "ibot_patch_loss.center"],
                    keys_not_sharded=["backbone.rope_embed.periods", "qkv.bias_mask"],
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")

    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)
        global_flip_labels = data.get("collated_global_flip_labels")
        if global_flip_labels is not None:
            global_flip_labels = global_flip_labels.cuda(non_blocking=True)
        local_side_labels = data.get("collated_local_side_labels")
        if local_side_labels is not None:
            local_side_labels = local_side_labels.cuda(non_blocking=True)
        local_crop_boxes = data.get("collated_local_crop_boxes")
        if local_crop_boxes is not None:
            local_crop_boxes = local_crop_boxes.cuda(non_blocking=True)
        local_source_global_indices = data.get("collated_local_source_global_indices")
        if local_source_global_indices is not None:
            local_source_global_indices = local_source_global_indices.cuda(non_blocking=True)

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Gram output
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            global_flip_labels=global_flip_labels,
            local_side_labels=local_side_labels,
            local_crop_boxes=local_crop_boxes,
            local_source_global_indices=local_source_global_indices,
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            iteration=iteration,
        )

        self.backprop_loss(loss_accumulator)

        # Return total weighted loss and a dict of metrics to log
        return loss_accumulator, metrics_dict | loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        if self.last_token_prior_enabled and self.last_token_prior_cls_mix > 0:
            importance = ibot_patch.norm(dim=-1)  # [n_crops * B, P]
            weights = torch.softmax(
                importance.float() / max(self.last_token_prior_temperature, 1e-6), dim=-1
            ).to(ibot_patch.dtype)
            last_cls = torch.bmm(weights.unsqueeze(1), ibot_patch).squeeze(1)  # [n_crops * B, D]
            cls = (1 - self.last_token_prior_cls_mix) * cls + self.last_token_prior_cls_mix * last_cls

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )  # [n_masked_patches, K]

        cls_pre_head = cls.unflatten(0, [n_crops, B])
        reg_pre_head = reg.unflatten(0, [n_crops, B])
        patch_pre_head = ibot_patch.unflatten(0, [n_crops, B])

        anatomy_prior, last_token_prior = self.get_teacher_anatomy_prior(
            patch_pre_head, H=H, W=W,
        )

        return {
            "cls_pre_head": cls_pre_head,  # [n_crops, B, D]
            "reg_pre_head": reg_pre_head,  # [n_crops, B, R, D]
            "patch_pre_head": patch_pre_head,  # [n_crops, B, P, D]
            "anatomy_prior": anatomy_prior,
            "last_token_prior": last_token_prior,
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    @torch.no_grad()
    def get_teacher_anatomy_prior(self, teacher_patch_tokens, *, H: int, W: int):
        """Returns (anatomy_prior, last_token_prior).

        LAST-ViT-inspired spatial-contrast prior is derived from the same L2 norms
        used by anatomy_discovery — no FFT, no extra large-tensor allocations,
        fully torch.compile-friendly.
        """
        if not self.anatomy_discovery_enabled:
            return None, None

        n_crops, B, P, D = teacher_patch_tokens.shape
        patch_h = H // self.cfg.student.patch_size
        patch_w = W // self.cfg.student.patch_size
        if P != patch_h * patch_w:
            logger.warning(
                "ANATOMY DISCOVERY: patch count mismatch P=%s, expected=%s; skipping teacher-guided prior",
                P,
                patch_h * patch_w,
            )
            return None, None

        if self.anatomy_discovery_source != "patch_l2":
            raise ValueError(
                f"Unsupported anatomy_discovery.source={self.anatomy_discovery_source}"
            )

        prior = teacher_patch_tokens.float().pow(2).sum(dim=-1).sqrt()

        last_token_prior = None
        if self.last_token_prior_enabled:
            k = self.last_token_prior_smooth_kernel
            pad = k // 2
            grid = prior.reshape(n_crops * B, 1, patch_h, patch_w)
            smoothed = torch.nn.functional.avg_pool2d(grid, kernel_size=k, stride=1, padding=pad)
            contrast = (grid / (grid - smoothed).abs().clamp_min(1e-6))
            contrast = contrast.reshape(n_crops, B, -1)
            contrast = _normalize_anatomy_prior(
                contrast,
                mode=self.last_token_prior_normalize,
                temperature=self.last_token_prior_temperature,
            )
            last_token_prior = contrast.reshape(n_crops, B, patch_h, patch_w)

        if self.anatomy_discovery_mix_with_static_prior > 0:
            ys = torch.linspace(0.0, 1.0, patch_h, device=prior.device, dtype=prior.dtype)
            xs = torch.linspace(0.0, 1.0, patch_w, device=prior.device, dtype=prior.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            gaussian_x = torch.exp(
                -0.5
                * ((xx - self.cfg.masking.focus_center_x) / max(self.cfg.masking.focus_spread_x, 1e-6)) ** 2
            )
            gaussian_y = torch.exp(
                -0.5
                * ((yy - self.cfg.masking.focus_center_y) / max(self.cfg.masking.focus_spread_y, 1e-6)) ** 2
            )
            static_prior = (1.0 + self.cfg.masking.bias_strength * gaussian_x * gaussian_y).reshape(1, 1, -1)
            prior = (1 - self.anatomy_discovery_mix_with_static_prior) * prior + (
                self.anatomy_discovery_mix_with_static_prior * static_prior
            )

        if (
            self.last_token_prior_enabled
            and self.last_token_prior_use_for_anatomy_discovery
            and last_token_prior is not None
        ):
            mix = float(getattr(self.cfg.anatomy_discovery, "last_prior_mix", 0.0))
            prior = _mix_patch_priors(
                prior.reshape(n_crops, B, -1),
                last_token_prior.reshape(n_crops, B, -1),
                mix,
            )

        prior = _normalize_anatomy_prior(
            prior,
            mode=self.anatomy_discovery_normalize,
            temperature=self.anatomy_discovery_temperature,
        )
        return prior.reshape(n_crops, B, patch_h, patch_w), last_token_prior

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

            with torch.no_grad():
                backbone_out = self.gram_teacher.backbone(images, is_training=True)
            teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

            # Downsample Gram teacher features if needed
            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.cfg.student.patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.cfg.student.patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(
                    -2, -1
                )  # [n_crops * B, N_student * N_student, D]
                assert teacher_patches.shape == student_patches.shape

        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_token_correspondence_loss(self, *, teacher_global, student_global, anatomy_prior, global_flip_labels):
        student_tokens = student_global["patch_pre_head"]
        teacher_tokens = teacher_global["patch_pre_head"]
        if self.token_correspondence_detach_teacher:
            teacher_tokens = teacher_tokens.detach()

        if self.token_correspondence_loss_type != "cosine":
            raise ValueError(
                f"Unsupported token_correspondence.loss_type={self.token_correspondence_loss_type}"
            )

        student_tokens = torch.nn.functional.normalize(student_tokens.float(), dim=-1)
        teacher_tokens = torch.nn.functional.normalize(teacher_tokens.float(), dim=-1)

        if not self.token_correspondence_mirror_aware:
            token_loss = 1.0 - (student_tokens * teacher_tokens).sum(dim=-1)
            if self.token_correspondence_use_teacher_prior and anatomy_prior is not None:
                weights = anatomy_prior.reshape(*anatomy_prior.shape[:2], -1).to(token_loss.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                return (token_loss * weights).sum(dim=-1).mean()
            return token_loss.mean()

        n_crops, batch_size, num_patches, _ = student_tokens.shape
        patch_h = patch_w = int(num_patches**0.5)
        if patch_h * patch_w != num_patches:
            logger.warning(
                "TOKEN CORRESPONDENCE: mirror-aware mode requires square patch grids, got P=%s; falling back",
                num_patches,
            )
            token_loss = 1.0 - (student_tokens * teacher_tokens).sum(dim=-1)
            return token_loss.mean()

        student_grid = student_tokens.reshape(n_crops, batch_size, patch_h, patch_w, -1)
        teacher_grid = teacher_tokens.reshape(n_crops, batch_size, patch_h, patch_w, -1)
        weight_grid = None
        if self.token_correspondence_use_teacher_prior and anatomy_prior is not None:
            weight_grid = anatomy_prior.to(student_grid.dtype)
            weight_grid = weight_grid / weight_grid.reshape(n_crops, batch_size, -1).sum(
                dim=-1, keepdim=True
            ).reshape(n_crops, batch_size, 1, 1).clamp_min(1e-6)
        if self.last_token_prior_enabled and self.last_token_prior_use_for_token_correspondence:
            last_prior = teacher_global.get("last_token_prior")
            if last_prior is not None:
                token_corr_mix = float(getattr(self.cfg.token_correspondence, "last_prior_mix", 0.0))
                last_prior = last_prior.to(student_grid.dtype)
                if weight_grid is None:
                    weight_grid = last_prior
                else:
                    mixed = _mix_patch_priors(
                        weight_grid.reshape(n_crops, batch_size, -1),
                        last_prior.reshape(n_crops, batch_size, -1),
                        token_corr_mix,
                    )
                    weight_grid = mixed.reshape(n_crops, batch_size, patch_h, patch_w)
                weight_grid = weight_grid / weight_grid.reshape(n_crops, batch_size, -1).sum(
                    dim=-1, keepdim=True
                ).reshape(n_crops, batch_size, 1, 1).clamp_min(1e-6)

        flip_labels_2d = None
        if self.token_correspondence_use_flip_labels and global_flip_labels is not None:
            flip_labels_2d = global_flip_labels.reshape(n_crops, batch_size)

        pair_losses = []
        for student_crop_idx in range(n_crops):
            for teacher_crop_idx in range(n_crops):
                if self.token_correspondence_cross_view_only and student_crop_idx == teacher_crop_idx:
                    continue
                aligned_teacher = teacher_grid[teacher_crop_idx]
                aligned_weights = None if weight_grid is None else weight_grid[teacher_crop_idx]
                if flip_labels_2d is not None:
                    mirrored_teacher = torch.flip(aligned_teacher, dims=(2,))
                    crop_flip_diff = (flip_labels_2d[student_crop_idx] != flip_labels_2d[teacher_crop_idx]).view(
                        batch_size, 1, 1, 1
                    )
                    aligned_teacher = torch.where(crop_flip_diff, mirrored_teacher, aligned_teacher)
                    if aligned_weights is not None:
                        mirrored_weights = torch.flip(aligned_weights, dims=(2,))
                        aligned_weights = torch.where(
                            crop_flip_diff.squeeze(-1), mirrored_weights, aligned_weights
                        )

                token_loss = 1.0 - (student_grid[student_crop_idx] * aligned_teacher).sum(dim=-1)
                if aligned_weights is None:
                    pair_losses.append(token_loss.mean())
                else:
                    pair_losses.append((token_loss * aligned_weights).sum(dim=(-1, -2)).mean())

        if pair_losses:
            return sum(pair_losses) / len(pair_losses)

        token_loss = 1.0 - (student_tokens * teacher_tokens).sum(dim=-1)
        return token_loss.mean()

    def compute_bandpass_consistency_loss(self, *, global_crops, teacher_global):
        n_global_crops, batch_size = global_crops.shape[:2]
        flat_global_crops = global_crops.flatten(0, 1)
        with torch.no_grad():
            low_crops = _frequency_filter_images(
                flat_global_crops,
                mode="low",
                cutoff=self.bandpass_low_cutoff,
                mean=self.cfg.crops.rgb_mean,
                std=self.cfg.crops.rgb_std,
            )
            high_crops = _frequency_filter_images(
                flat_global_crops,
                mode="high",
                cutoff=self.bandpass_high_cutoff,
                mean=self.cfg.crops.rgb_mean,
                std=self.cfg.crops.rgb_std,
            )

        teacher_cls = teacher_global["cls_pre_head"].flatten(0, 1).detach()
        teacher_patch = teacher_global["patch_pre_head"].flatten(0, 1).detach()
        teacher_cls = torch.nn.functional.normalize(teacher_cls.float(), dim=-1)
        teacher_patch = torch.nn.functional.normalize(teacher_patch.float(), dim=-1)

        # Run low/high filtered crops in two passes instead of concatenating them.
        # This keeps the auxiliary bandpass branch from doubling peak activation memory.
        low_out = self.student.backbone(low_crops, is_training=True)
        low_cls = torch.nn.functional.normalize(low_out["x_norm_clstoken"].float(), dim=-1)
        low_cls_loss = 1.0 - (low_cls * teacher_cls).sum(dim=-1).mean()

        del low_out, low_cls, low_crops

        high_out = self.student.backbone(high_crops, is_training=True)
        high_patch = torch.nn.functional.normalize(high_out["x_norm_patchtokens"].float(), dim=-1)
        high_patch_loss = 1.0 - (high_patch * teacher_patch).sum(dim=-1).mean()

        del high_out, high_patch, high_crops

        bandpass_loss = self.bandpass_cls_loss_weight * low_cls_loss + self.bandpass_patch_loss_weight * high_patch_loss
        return bandpass_loss, {
            "bandpass_low_cls_loss": low_cls_loss,
            "bandpass_high_patch_loss": high_patch_loss,
            "bandpass_batch_size": torch.as_tensor(
                n_global_crops * batch_size,
                device=global_crops.device,
                dtype=bandpass_loss.dtype,
            ),
        }

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        global_flip_labels,
        local_side_labels,
        local_crop_boxes,
        local_source_global_indices,
        global_crops,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        anatomy_prior = teacher_global.get("anatomy_prior")
        last_token_prior = teacher_global.get("last_token_prior")
        if anatomy_prior is not None:
            loss_dict["anatomy_prior_mean"] = anatomy_prior.mean()
            loss_dict["anatomy_prior_max"] = anatomy_prior.max()
        if last_token_prior is not None:
            loss_dict["last_token_prior_mean"] = last_token_prior.mean()
            loss_dict["last_token_prior_max"] = last_token_prior.max()
        if self.token_correspondence_enabled:
            token_correspondence_loss = self.compute_token_correspondence_loss(
                teacher_global=teacher_global,
                student_global=student_global,
                anatomy_prior=anatomy_prior,
                global_flip_labels=global_flip_labels,
            )
            loss_accumulator += self.token_correspondence_loss_weight * token_correspondence_loss
            loss_dict["token_correspondence_loss"] = token_correspondence_loss
            loss_dict["token_correspondence_loss_weight"] = self.token_correspondence_loss_weight

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        if self.bandpass_consistency_enabled:
            bandpass_loss, bandpass_metrics = self.compute_bandpass_consistency_loss(
                global_crops=global_crops,
                teacher_global=teacher_global,
            )
            loss_accumulator += self.bandpass_consistency_loss_weight * bandpass_loss
            loss_dict["bandpass_consistency_loss"] = bandpass_loss
            loss_dict["bandpass_consistency_loss_weight"] = self.bandpass_consistency_loss_weight
            loss_dict.update(bandpass_metrics)

        if self.flip_prediction_enabled and global_flip_labels is not None:
            flip_logits = self.student.flip_head(student_global["cls_pre_head"].flatten(0, 1)).squeeze(-1)
            flip_loss = self.flip_prediction_loss(flip_logits, global_flip_labels)
            flip_accuracy = ((flip_logits > 0).float() == global_flip_labels).float().mean()
            loss_accumulator += self.flip_prediction_loss_weight * flip_loss
            loss_dict["flip_prediction_loss"] = flip_loss
            loss_dict["flip_prediction_acc"] = flip_accuracy

        if self.local_side_prediction_enabled and local_side_labels is not None:
            local_side_logits = self.student.local_side_head(student_local["cls_pre_head"].flatten(0, 1))
            local_side_loss = self.local_side_prediction_loss(local_side_logits, local_side_labels)
            local_side_acc = (local_side_logits.argmax(dim=-1) == local_side_labels).float().mean()
            loss_accumulator += self.local_side_prediction_loss_weight * local_side_loss
            loss_dict["local_side_prediction_loss"] = local_side_loss
            loss_dict["local_side_prediction_acc"] = local_side_acc

        # --- Flip CLS Symmetry Loss ---
        # For each pair of global crops (crop_i, crop_j, i≠j) from the same image:
        #   student(crop_i) should align with teacher(crop_j) in dense embedding space.
        # Pairs whose flip labels DIFFER (anatomically mirrored) receive an extra weight,
        # making the model more sensitive to bilateral dental symmetry.
        if self.flip_cls_symmetry_enabled and global_flip_labels is not None:
            import torch.nn.functional as F_local
            n_g = student_global["cls_pre_head"].shape[0]   # n_global_crops
            B_sz = student_global["cls_pre_head"].shape[1]
            flip_labels_2d = global_flip_labels.reshape(n_g, B_sz)  # [n_crops, B]
            sym_loss_accum = 0.0
            sym_pair_count = 0
            for ci in range(n_g):
                for cj in range(n_g):
                    if ci == cj:
                        continue
                    s_i = F_local.normalize(
                        student_global["cls_pre_head"][ci].float(), dim=-1
                    )  # [B, D]
                    t_j = F_local.normalize(
                        teacher_global["cls_pre_head"][cj].float().detach(), dim=-1
                    )  # [B, D]
                    # 1 - cosine similarity per sample
                    pair_loss = 1.0 - (s_i * t_j).sum(dim=-1)  # [B]
                    # Upweight pairs with opposite flip labels (true bilateral-symmetric views)
                    flip_diff = (flip_labels_2d[ci] != flip_labels_2d[cj]).float()  # [B]
                    weights = 1.0 + self.flip_cls_symmetry_flip_bonus * flip_diff
                    sym_loss_accum += (pair_loss * weights).mean()
                    sym_pair_count += 1
            sym_loss = sym_loss_accum / max(sym_pair_count, 1)
            loss_accumulator += self.flip_cls_symmetry_loss_weight * sym_loss
            loss_dict["flip_cls_symmetry_loss"] = sym_loss
            loss_dict["flip_cls_symmetry_loss_weight"] = self.flip_cls_symmetry_loss_weight

        if self.mim_enabled:
            # TRACE_STAGEA_S3_MIM: regress normalized patch pixels for masked global tokens only.
            mim_predictions = self.student.mim_head(student_global["masked_patch_pre_head"])
            mim_targets_raw = _patchify_for_stage_a(
                global_crops.flatten(0, 1),
                patch_size=self.cfg.student.patch_size,
            )
            mim_targets_raw = torch.index_select(mim_targets_raw.flatten(0, 1), dim=0, index=mask_indices_list)
            mim_targets = mim_targets_raw
            if self.mim_normalize_targets:
                target_mean = mim_targets.mean(dim=-1, keepdim=True)
                target_var = mim_targets.var(dim=-1, keepdim=True, unbiased=False)
                mim_targets = (mim_targets - target_mean) / (target_var + 1.0e-6).sqrt()
            if mim_targets.shape[0] == 0:
                mim_loss = mim_predictions.sum() * 0.0
            elif (
                self.mim_variant == "pixel"
                and self.mim_edge_loss_weight <= 0
                and self.mim_transition_weight <= 0
            ):
                mim_loss = torch.nn.functional.mse_loss(mim_predictions, mim_targets)
            else:
                pred_patches = _tokens_to_patch_images(
                    mim_predictions,
                    patch_size=self.cfg.student.patch_size,
                    channels=self.cfg.student.in_chans,
                )
                target_patches = _tokens_to_patch_images(
                    mim_targets,
                    patch_size=self.cfg.student.patch_size,
                    channels=self.cfg.student.in_chans,
                )
                structure_target_patches = _tokens_to_patch_images(
                    mim_targets_raw,
                    patch_size=self.cfg.student.patch_size,
                    channels=self.cfg.student.in_chans,
                )
                transition_weights, transition_energy = _build_transition_weights(
                    structure_target_patches,
                    transition_weight=self.mim_transition_weight,
                    transition_gamma=self.mim_transition_gamma,
                )
                pixel_loss_per_patch = torch.nn.functional.mse_loss(
                    pred_patches,
                    target_patches,
                    reduction="none",
                ).mean(dim=(1, 2, 3))
                weighted_pixel_loss = (pixel_loss_per_patch * transition_weights).mean()
                mim_loss = weighted_pixel_loss
                loss_dict["mim_transition_weight_mean"] = transition_weights.mean()
                loss_dict["mim_transition_energy_mean"] = transition_energy.mean()
                if self.mim_variant == "boundary_transition_aware":
                    pred_grad = _sobel_gradient_magnitude(pred_patches)
                    target_grad = _sobel_gradient_magnitude(target_patches)
                    edge_loss_per_patch = torch.nn.functional.l1_loss(
                        pred_grad,
                        target_grad,
                        reduction="none",
                    ).mean(dim=(1, 2, 3))
                    edge_loss = (edge_loss_per_patch * transition_weights).mean()
                    mim_loss = mim_loss + self.mim_edge_loss_weight * edge_loss
                    loss_dict["mim_edge_loss"] = edge_loss
            loss_accumulator += self.mim_loss_weight * mim_loss
            loss_dict["mim_loss"] = mim_loss
            loss_dict["mim_loss_weight"] = self.mim_loss_weight

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    # Save stats over masked / unmasked tokens
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg):
        augmentation_cfg = getattr(cfg, "augmentation", None)
        anatomy_guided_masking_cfg = getattr(cfg, "anatomy_guided_masking", None)
        return DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            global_crops_ratio=(
                tuple(cfg.crops.global_crops_ratio) if "global_crops_ratio" in cfg.crops else None
            ),
            teacher_no_color_jitter=(
                cfg.crops.teacher_no_color_jitter if "teacher_no_color_jitter" in cfg.crops else False
            ),
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
            output_channels=cfg.student.in_chans,
            medical_augmentation=augmentation_cfg.medical_augmentation if augmentation_cfg is not None else False,
            gamma_range=tuple(augmentation_cfg.gamma_range) if augmentation_cfg is not None else (0.8, 1.2),
            contrast_range=tuple(augmentation_cfg.contrast_range) if augmentation_cfg is not None else (0.85, 1.15),
            brightness_range=tuple(augmentation_cfg.brightness_range) if augmentation_cfg is not None else (0.9, 1.1),
            rotation_range=tuple(augmentation_cfg.rotation_range) if augmentation_cfg is not None else (0, 5),
            augmentation_mode=augmentation_cfg.mode if augmentation_cfg is not None else "natural",
            local_crop_strategy=cfg.crops.local_crop_strategy,
            direct_center_jitter=(
                cfg.crops.direct_center_jitter if "direct_center_jitter" in cfg.crops else 0.03
            ),
            direct_random_scale=(
                cfg.crops.direct_random_scale if "direct_random_scale" in cfg.crops else True
            ),
            local_quadrant_ratio=cfg.crops.local_quadrant_ratio,
            local_arch_focus_strength=cfg.crops.local_arch_focus_strength,
            local_crop_jitter=cfg.crops.local_crop_jitter,
            local_crop_min_scale=cfg.crops.local_crop_min_scale,
            local_crop_max_scale=cfg.crops.local_crop_max_scale,
            band_aware_fast_crop_min_ratio=(
                cfg.crops.band_aware_fast_crop_min_ratio
                if "band_aware_fast_crop_min_ratio" in cfg.crops
                else 0.22
            ),
            band_aware_fast_crop_max_ratio=(
                cfg.crops.band_aware_fast_crop_max_ratio
                if "band_aware_fast_crop_max_ratio" in cfg.crops
                else 0.31
            ),
            band_aware_fast_upper_center_offset=(
                cfg.crops.band_aware_fast_upper_center_offset
                if "band_aware_fast_upper_center_offset" in cfg.crops
                else 0.09
            ),
            band_aware_fast_lower_center_offset=(
                cfg.crops.band_aware_fast_lower_center_offset
                if "band_aware_fast_lower_center_offset" in cfg.crops
                else 0.09
            ),
            band_aware_fast_use_official_area_scale=(
                cfg.crops.band_aware_fast_use_official_area_scale
                if "band_aware_fast_use_official_area_scale" in cfg.crops
                else False
            ),
            representative_tooth_aware_attempts=(
                cfg.crops.representative_tooth_aware_attempts
                if "representative_tooth_aware_attempts" in cfg.crops
                else 80
            ),
            representative_tooth_cache_path=(
                cfg.crops.representative_tooth_cache_path
                if "representative_tooth_cache_path" in cfg.crops
                else None
            ),
            xray_noise_probability=augmentation_cfg.xray_noise_probability if augmentation_cfg is not None else 0.15,
            xray_noise_std=tuple(augmentation_cfg.xray_noise_std) if augmentation_cfg is not None else (0.005, 0.02),
            anatomy_guided_masking_enabled=(
                anatomy_guided_masking_cfg.enabled if anatomy_guided_masking_cfg is not None else False
            ),
            anatomy_guided_masking_source=(
                anatomy_guided_masking_cfg.source if anatomy_guided_masking_cfg is not None else "image_intensity"
            ),
            anatomy_guided_masking_mix_with_static_prior=(
                anatomy_guided_masking_cfg.mix_with_static_prior if anatomy_guided_masking_cfg is not None else 0.5
            ),
            anatomy_guided_masking_epsilon=(
                anatomy_guided_masking_cfg.epsilon if anatomy_guided_masking_cfg is not None else 1e-6
            ),
            layered_masking_enabled=(
                anatomy_guided_masking_cfg.layered_enabled if anatomy_guided_masking_cfg is not None else False
            ),
            layered_mask_core_weight=(
                anatomy_guided_masking_cfg.layered_core_weight if anatomy_guided_masking_cfg is not None else 0.55
            ),
            layered_mask_context_weight=(
                anatomy_guided_masking_cfg.layered_context_weight if anatomy_guided_masking_cfg is not None else 0.30
            ),
            layered_mask_background_weight=(
                anatomy_guided_masking_cfg.layered_background_weight
                if anatomy_guided_masking_cfg is not None
                else 0.15
            ),
            anatomy_guided_masking_full_prior_max_side=(
                anatomy_guided_masking_cfg.full_prior_max_side
                if anatomy_guided_masking_cfg is not None and "full_prior_max_side" in anatomy_guided_masking_cfg
                else 512
            ),
            anatomy_guided_masking_projection=(
                OmegaConf.to_container(anatomy_guided_masking_cfg.projection, resolve=True)
                if anatomy_guided_masking_cfg is not None and "projection" in anatomy_guided_masking_cfg
                else None
            ),
            anatomy_guided_masking_tooth_prior=(
                OmegaConf.to_container(anatomy_guided_masking_cfg.tooth_prior, resolve=True)
                if anatomy_guided_masking_cfg is not None and "tooth_prior" in anatomy_guided_masking_cfg
                else None
            ),
            global_crop_strategy=(
                cfg.crops.global_crop_strategy if "global_crop_strategy" in cfg.crops else "random"
            ),
            global_tooth_center_jitter=(
                cfg.crops.global_tooth_center_jitter if "global_tooth_center_jitter" in cfg.crops else 0.035
            ),
            global_tooth_center_scale=(
                tuple(cfg.crops.global_tooth_center_scale)
                if "global_tooth_center_scale" in cfg.crops
                else (0.56, 0.60)
            ),
            hierarchical_transition_ratio=(
                cfg.crops.hierarchical_transition_ratio if "hierarchical_transition_ratio" in cfg.crops else 0.0
            ),
            view_aware_augmentation=(
                augmentation_cfg.view_aware_enabled if augmentation_cfg is not None and "view_aware_enabled" in augmentation_cfg else False
            ),
            global_jitter_scale=(
                augmentation_cfg.global_jitter_scale if augmentation_cfg is not None and "global_jitter_scale" in augmentation_cfg else 0.75
            ),
            local_jitter_scale=(
                augmentation_cfg.local_jitter_scale if augmentation_cfg is not None and "local_jitter_scale" in augmentation_cfg else 1.15
            ),
            global_noise_scale=(
                augmentation_cfg.global_noise_scale if augmentation_cfg is not None and "global_noise_scale" in augmentation_cfg else 0.75
            ),
            local_noise_scale=(
                augmentation_cfg.local_noise_scale if augmentation_cfg is not None and "local_noise_scale" in augmentation_cfg else 1.25
            ),
            blur_probability_global1=(
                augmentation_cfg.blur_probability_global1
                if augmentation_cfg is not None and "blur_probability_global1" in augmentation_cfg
                else None
            ),
            blur_probability_global2=(
                augmentation_cfg.blur_probability_global2
                if augmentation_cfg is not None and "blur_probability_global2" in augmentation_cfg
                else None
            ),
            blur_probability_local=(
                augmentation_cfg.blur_probability_local
                if augmentation_cfg is not None and "blur_probability_local" in augmentation_cfg
                else None
            ),
            spectral_augmentation_enabled=(
                augmentation_cfg.spectral_enabled if augmentation_cfg is not None and "spectral_enabled" in augmentation_cfg else False
            ),
            spectral_augmentation_p=(
                augmentation_cfg.spectral_p if augmentation_cfg is not None and "spectral_p" in augmentation_cfg else 0.5
            ),
            spectral_mask_low_range=tuple(
                augmentation_cfg.spectral_mask_low_range
                if augmentation_cfg is not None and "spectral_mask_low_range" in augmentation_cfg
                else (0.08, 0.40)
            ),
            spectral_mask_width_range=tuple(
                augmentation_cfg.spectral_mask_width_range
                if augmentation_cfg is not None and "spectral_mask_width_range" in augmentation_cfg
                else (0.10, 0.30)
            ),
            spectral_attenuation_range=tuple(
                augmentation_cfg.spectral_attenuation_range
                if augmentation_cfg is not None and "spectral_attenuation_range" in augmentation_cfg
                else (0.30, 0.85)
            ),
            local_policy=(
                OmegaConf.to_container(cfg.crops.local_policy, resolve=True)
                if "local_policy" in cfg.crops and cfg.crops.local_policy is not None
                else None
            ),
        )

    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
            inference_only_models_process_groups.append(default_process_group)
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def broadcast_to_subgroups(self, tensor, over_dim, global_batch_size=None):
        """
        This is an operation that takes a tensor from the default process group, gathers it, stacks it, then scatters it within a smaller process subgroup
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
