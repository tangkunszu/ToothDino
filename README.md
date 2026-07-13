# ToothDINO

ToothDINO is a geometry-aware continued-pretraining recipe for panoramic dental radiographs. It starts from the official DINOv3 ViT-B/16 checkpoint and keeps the DINOv3 backbone and self-supervised losses unchanged. The dental adaptation is introduced through annotation-free view construction and mask sampling:

- **DAVC: Dental-Aware View Construction** builds two panorama-level global views and eight tooth-centric local views.
- **DXA: Dental X-ray Augmentation** replaces natural-image color augmentation with grayscale radiograph-compatible intensity and geometric perturbations.
- **TCC: Tooth-Centric Cropping** places local crops around the tooth-bearing region using a weak edge-intensity geometric prior, without tooth boxes or segmentation masks.
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
| local crop scale after tooth-centric center selection | `[0.65, 1.0]` |
| local crop jitter | `tau_x = tau_y = 0.15` |

The current implementation entry point is:

- `dinov3/data/augmentations.py`
- `dinov3/data/augmentations_medical.py`

Relevant implementation symbols:

- `DataAugmentationDINO`
- `MedicalImageAugmentation`
- `_sample_global_crop_with_meta`
- `_sample_hierarchical_local_crops`

## DXA: Dental X-ray Augmentation

DXA avoids hue, saturation, and solarization because they do not have a meaningful grayscale radiograph counterpart.

Paper settings:

| Transform | Range |
| --- | --- |
| gamma | `[0.8, 1.2]` |
| contrast | `[0.85, 1.15]` |
| brightness | `[0.9, 1.1]` |
| in-plane rotation | `[0, 5]` degrees |
| local Gaussian noise std | `[0.005, 0.02]` |

The local branch uses stronger crop-level perturbation than the global branch.

## TCC: Tooth-Centric Cropping

TCC estimates a weak dental band prior from image content. It does not use labels.

The paper defines an edge-intensity saliency map:

```text
E(u, v) = |grad_u x(u, v)| + 0.75 * |grad_v x(u, v)|
S(u, v) = E(u, v) * (0.35 + 0.65 * psi(x(u, v))) * pi_v(v)
psi(z) = clip((z - 0.18) / 0.52, 0, 1)
```

The implementation then estimates the dental extent using smoothed horizontal and vertical projections and distributes representative crop centers from left to right along upper and lower tooth rows.

Paper thresholds:

| Quantity | Value |
| --- | --- |
| horizontal energy threshold | `0.48 * max smoothed column energy` |
| row energy threshold | `0.58 * max smoothed row energy within estimated extent` |
| center jitter | `U(-0.15, 0.15)` in normalized coordinates |

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
| center `(c_x, c_y)` | `(0.5, 0.5)` |
| spread `(sigma_x, sigma_y)` | `(0.35, 0.20)` |
| mask ratio | `U(0.1, 0.5)` |

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
| unlabeled corpus | 57,232 panoramic dental radiographs |
| optimizer | AdamW |
| base learning rate | `1e-4` |
| batch size | `64` per GPU |
| epochs | `200` |
| warm-up | `10` epochs |
| weight decay | `0.04 -> 0.4` |
| LR schedule | cosine |
| teacher update | EMA |

Current repository note: `dinov3/configs/train/vitb_toothdino_lowrisk.yaml` currently sets `optim.epochs: 300`. For strict paper reproduction, set it to `200`.

## Main Config

Use this config as the paper-aligned ToothDINO entry point:

```bash
dinov3/configs/train/vitb_toothdino_paper.yaml
```

The older local experiment config is also available:

```bash
dinov3/configs/train/vitb_toothdino_lowrisk.yaml
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
    train/vitb_toothdino_lowrisk.yaml
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
