# ToothDINO

ToothDINO is a geometry-aware continued-pretraining recipe for panoramic dental radiographs. It starts from the official DINOv3 ViT-B/16 checkpoint and keeps the DINOv3 backbone and self-supervised losses unchanged. The dental adaptation is introduced through annotation-free view construction and mask sampling:

- **DAVC: Dental-Aware View Construction** builds two panorama-level global views and eight tooth-centric local views.
- **DXA: Dental X-ray Augmentation** replaces natural-image color augmentation with grayscale radiograph-compatible intensity and geometric perturbations.
- **TCC: Tooth-Centric Cropping** places local crops around representative tooth-row centers using a weak edge-intensity geometric prior, without tooth boxes or segmentation masks.
- **ABM: Anatomy-Biased Masking** samples iBOT masked-token positions with a fixed elliptical dental-band prior on the global-view patch grid.

This document is extracted from `newtoothdino.tex` and aligned with the current implementation in this repository.

## Method Summary

ToothDINO follows the DINOv3 student-teacher framework. The teacher receives two unmasked global views. The student receives the corresponding masked global views and all tooth-centric local views.

The continued-pretraining objective is unchanged:

```text
L_ToothDINO = L_DINO + lambda_iBOT * L_iBOT + lambda_KoLeo * L_KoLeo
```

The paper uses:

- `lambda_iBOT = 1.0`
- `lambda_KoLeo = 0.1`
- no Gram-anchoring refinement stage during dental continued pretraining

## Dental-Aware View Construction

For each panoramic radiograph:

1. Apply shared medical augmentation `T_med`.
2. Sample two global crops.
3. Sample eight tooth-centric local crops.
4. Apply crop-level DXA to global and local branches.

Paper settings:

| Component | Setting |
| --- | --- |
| global views | 2 |
| local views | 8 |
| global resolution | `256 x 256` |
| local resolution | `128 x 128` |
| global crop scale | `[0.32, 1.0]` |
| local crop area | `local_crops_scale = [0.05, 0.26]`, sampled per view, anchored to `H^2` |
| local crop center | representative upper/lower tooth-row centers, with bounded jitter |
| paper strategy | `n_tcc` (legacy alias: `dental_representative_tooth_direct`) |

The current implementation entry point is:

- `dinov3/data/augmentations.py`
- `dinov3/data/augmentations_medical.py`

Relevant implementation symbols:

- `DataAugmentationDINO`
- `MedicalImageAugmentation`
- `_sample_global_crop_with_meta`
- `_sample_representative_tooth_direct_local_crops`
- `_sample_fixed_center_local_crops`
- `_get_representative_tooth_centers_fast`
- `_representative_tooth_centers_from_band`

## DXA: Dental X-ray Augmentation

DXA avoids hue, saturation, and solarization because they do not have a meaningful grayscale radiograph counterpart.

Paper settings:

| Transform | Range |
| --- | --- |
| gamma | `[0.8, 1.2]` |
| contrast | `[0.85, 1.15]` |
| brightness | `[0.9, 1.1]` |
| in-plane rotation | `[0, 5]` degrees |
| Gaussian noise std (both branches) | `[0.005, 0.02]` |
| jitter strength, global / local | `0.75` / `1.15` |
| noise probability, global / local | `0.1125` / `0.1875` (base `0.15` scaled by `0.75` / `1.25`) |
| blur probability, global view 1 / 2 | `0.30` / `0.10` |
| blur probability, local | `0.40` |

The two crop-level stages share the same transformation family and differ only in
strength and application probability: the noise standard deviation range is identical in
both branches, and the local branch is perturbed more strongly and more often. The blur
probabilities are set explicitly because the inherited DINOv3 defaults
(`0.45 / 0.15 / 0.40`) make global view 1 blur *more* often than the local branch, which
inverts the weak-global / strong-local intent. The asymmetry between the two global views
is retained so that the two teacher targets differ, which is a separate purpose.

## TCC: Tooth-Centric Cropping

TCC estimates a weak dental band prior from image content. It does not use labels.

The paper defines an edge-intensity saliency map:

```text
E(u, v) = |grad_u x(u, v)| + 0.75 * |grad_v x(u, v)|
S(u, v) = E(u, v) * (0.35 + 0.65 * psi(x(u, v))) * pi_v(v)
psi(z) = clip((z - 0.18) / 0.52, 0, 1)
```

The paper configuration uses `n_tcc`: the implementation estimates the dental extent using
smoothed horizontal and vertical projections, distributes representative crop centers from
left to right along upper and lower tooth rows, and samples local crops around those
centers with bounded center jitter and per-view area and aspect-ratio sampling. This
representative-center policy trades generic crop-position entropy for anatomically
concentrated local supervision, while the jitter and scale sampling keep the local views
from being pixel-identical across epochs.

Local crop area is anchored to `H^2` rather than to the full image area `W*H`. Panoramic
radiographs have an aspect ratio close to `2:1`, so a crop sized from the full image area
comes out about `sqrt(W/H) = 1.41` times taller than the standard local-crop semantics
intend, up to `0.80 * H`, at which point the crop box is clamped back inside the image and
the lower tooth row's crops are dragged off their anchors toward the upper row. With the
`0.26` upper bound the tallest crop stays at `0.589 * H`, below that threshold.

`dental_representative_tooth_direct` and `dental_representative_tooth_direct_jitter` are
retained as legacy aliases of `n_tcc` and select the same code path. Setting
`direct_center_jitter: 0.0` and `direct_random_scale: false` recovers the deterministic
variant used for ablation.

Paper thresholds:

| Quantity | Value |
| --- | --- |
| horizontal energy threshold | `0.48 * max smoothed column energy` |
| row energy threshold | `0.58 * max smoothed row energy within estimated extent` |
| representative centers | four upper-row and four lower-row anchors |
| center jitter | `direct_center_jitter = 0.03`, i.e. `U(-0.03, 0.03)` of image width and height |
| local crop area | `U(0.05, 0.26)` of `H^2`, aspect ratio `exp(U(log(3/4), log(4/3)))` |
| deterministic ablation | `direct_center_jitter = 0.0`, `direct_random_scale = false` |

## ABM: Anatomy-Biased Masking

ABM affects only the student global branch. The teacher global branch remains unmasked.

For a patch coordinate `(u_a, v_b)` on the global-view grid:

```text
d_ab^2 = ((u_a - c_x)^2 / sigma_x^2) + ((v_b - c_y)^2 / sigma_y^2)
w_ab = 1 + beta * exp(-0.5 * d_ab^2)
p_ab = w_ab / sum(w)
```

Paper settings:

| Parameter | Value |
| --- | --- |
| `beta` | `2.5` |
| center `(c_x, c_y)` | `(0.5, 0.61)` |
| spread `(sigma_x, sigma_y)` | `(0.40, 0.25)` |
| mask ratio range | `[0.1, 0.5]` |
| masked fraction of student global views | `0.5` (`ibot.mask_sample_probability`) |

The vertical center sits below the view center because the prior lives on the global-view
patch grid, and global views are random resized crops, so the tooth-bearing band does not
coincide with the view center. Measured over the full pretraining corpus -- 57,090
radiographs, 114,180 global views, and 736,074 tooth anchors visible inside a view --
anchors land at `x = 0.500 +- 0.246` and `y = 0.610 +- 0.128` in normalized view
coordinates. The uncalibrated `(0.5, 0.5)` / `(0.35, 0.20)` prior puts `0.610` of its mass
on real tooth anchors; the values above raise that to `0.749`.

Both sit at their optimum. Sweeping `c_y` at fixed spread peaks on a plateau at
`0.60 - 0.62` (`0.50 -> 0.696`, `0.56 -> 0.738`, `0.60 -> 0.749`, `0.62 -> 0.749`,
`0.66 -> 0.737`, `0.70 -> 0.712`). On a `0.05` grid no spread of smaller area than
`0.40 x 0.25` reaches this mass (`(0.45, 0.20) -> 0.730`, `(0.50, 0.20) -> 0.748`,
`(0.35, 0.25) -> 0.716`); wider spreads raise it further (`(0.55, 0.30) -> 0.836`) but
only by flattening the prior toward uniform. Restricting the measurement to images with
aspect ratio `>= 1.4` (48,384 radiographs, 602,901 anchors) changes nothing material
(`0.609 -> 0.748`). Regenerate with `calibrate_abm_prior.py`.

Two implementation details are worth stating explicitly, because both apply identically to
the uniform-masking control and therefore do not confound the comparison:

1. Only a fraction `0.5` of the student global views in each batch are masked; the rest
   receive an empty mask and do not contribute to the iBOT term.
2. Masking ratios are not drawn i.i.d. from `U(0.1, 0.5)`. For the `n` masked views in a
   batch, the ratios are a deterministic stratification of `[0.1, 0.5]`, so each batch
   covers the whole range exactly once; the assignment of ratios to views is shuffled.
3. Masks are assembled from rectangular patch blocks, as in iBOT and DINOv3, not from
   independently drawn patches. Block area is `U(4, M)` patches with log-uniform aspect
   ratio in `[0.3, 10/3]`, and each block's top-left corner is drawn with probability
   proportional to the mean of `p` over the window it would cover. Any shortfall is filled
   by weighted sampling without replacement from `p`. The uniform control replaces `p` by
   the uniform distribution in both steps and leaves everything else unchanged.

Implementation entries:

- `dinov3/data/masking.py`
- `dinov3/data/collate.py`
- `dinov3/train/train.py`

Relevant symbols:

- `MaskingGenerator`
- `masking.bias_mode: anatomy_biased`
- `collated_mask_bias_maps` for optional image-conditioned masking variants, not required for the fixed paper ABM prior

## Pretraining Recipe

Paper setting:

| Item | Value |
| --- | --- |
| initialization | official DINOv3 ViT-B/16 weights |
| backbone | ViT-B/16 |
| patch size | 16 |
| unlabeled corpus | 57,090 panoramic dental radiographs |
| optimizer | AdamW |
| base learning rate | `1e-4` |
| LR scaling rule | `sqrt_wrt_1024` |
| peak learning rate | `5e-5` (`1e-4 * sqrt(256/1024)`) |
| minimum learning rate | `1e-6` |
| batch size | `64` per GPU, `256` global on 4 GPUs |
| epochs | `200` (`OFFICIAL_EPOCH_LENGTH = 223`, 44,600 steps) |
| warm-up | `10` epochs |
| weight decay | `0.04 -> 0.4` |
| LR schedule | cosine |
| layer-wise LR decay | `0.9` |
| patch-embed LR multiplier | `0.2` |
| gradient clipping | `3.0` |
| freeze last layer | first `2` epochs |
| teacher update | EMA, momentum `0.996 -> 1.0` |
| teacher temperature | `0.04 -> 0.07` over `10` epochs |
| centering | Sinkhorn-Knopp |
| register tokens | `4` |
| precision | bf16 parameters, fp32 gradient reduction |

## Main Config

Use this config as the paper-aligned ToothDINO entry point:

```bash
dinov3/configs/train/vitb_toothdino_paper.yaml
```

Before training, set the official DINOv3-B/16 checkpoint and dataset path:

```yaml
MODEL:
  WEIGHTS: /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
student:
  pretrained_weights: /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
train:
  dataset_path: MyDINOSet:root=/path/to/panoramic_xray_train/
  output_dir: /path/to/output/toothdino
```

Example launch:

```bash
torchrun --nproc_per_node=4 -m dinov3.train.train \
  --config-file dinov3/configs/train/vitb_toothdino_paper.yaml \
  MODEL.WEIGHTS=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  student.pretrained_weights=/path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  train.dataset_path=MyDINOSet:root=/path/to/panoramic_xray_train/ \
  train.output_dir=./output/toothdino_paper
```

## Downstream Evaluation Protocol

The paper evaluates ToothDINO on seven datasets and ten settings across:

- classification
- object detection
- instance segmentation
- semantic segmentation

Paper downstream protocol:

| Task | Framework | Head / Model | Input | Batch | Schedule |
| --- | --- | --- | --- | --- | --- |
| classification | MMPreTrain | linear head | `512 x 512` | 16 | 50 epochs |
| detection / instance segmentation | MMDetection | Mask R-CNN + FPN | `1333 x 800` | 4 | 12 epochs |
| semantic segmentation | MMSegmentation | FPN + UPerNet-style decoder + auxiliary FCN head | `1024 x 1024` | 4 | 20k iterations |

The controlled comparison uses matched rank-8 LoRA adaptation for DINOv3-B-CPT and ToothDINO.

## Minimal Files To Publish

For a clean GitHub release, keep at least:

```text
dinov3/
  configs/
    ssl_default_config.yaml
    train/vitb_toothdino_paper.yaml
  data/
    augmentations.py
    augmentations_medical.py
    collate.py
    masking.py
  train/
    train.py
    ssl_meta_arch.py
requirements-toothdino.txt
environment-toothdino.yml
docs/TOOTHDINO_GITHUB_README.md
```

Optional but useful:

```text
tools/visualize_a_multiscale_crops.py
tools/visualize_opg_masking.py
tools/generate_toothdino_framework_v2.py
downstream_tasks/
```

Do not publish local outputs, logs, private checkpoints, institutional data paths, temporary slide assets, or generated experiment directories unless they are intentionally anonymized.

## Version Requirements

The repository declares `python_requires >= 3.11`. The current local environment used for this extraction has:

```text
python >= 3.11
torch == 2.7.1+cu118
torchvision == 0.22.1+cu118
torchmetrics == 1.6.1
omegaconf == 2.3.0
numpy == 1.26.4
Pillow == 11.3.0
scikit-learn == 1.6.1
ftfy == 6.3.1
regex == 2.5.148
mmengine == 0.10.4
```

OpenMMLab packages are required only for downstream evaluation. Install versions compatible with your CUDA/PyTorch stack, for example:

```text
mmcv >= 2.0.0
mmengine >= 0.10.0
mmdet >= 3.0.0
mmsegmentation >= 1.0.0
mmpretrain >= 1.0.0
```

Use `requirements-toothdino.txt` for the pretraining stack and `environment-toothdino.yml` if you prefer conda.

## Citation

This repository builds on DINOv3. Please cite the original DINOv3 work:

```bibtex
@article{simeoni2025dinov3,
  title={DINOv3},
  author={Simeoni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}
```
