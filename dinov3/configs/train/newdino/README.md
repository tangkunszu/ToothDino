# newdino ablation configs

Two ablation axes, matching the paper's two dental-aware modules:

| Module | Contains | xiaorong shorthand |
| --- | --- | --- |
| **DAVC** — Dental-Aware View Construction | radiograph-consistent augmentation (no hue/saturation/solarize) + view-aware strength (weak global / strong local) + n-TCC tooth-centric local views | `A` + `B` |
| **ABM** — Anatomy-Biased Masking | anatomy-biased iBOT masked-token sampling | `C` |

The view-aware weak/strong scaling lives **inside DAVC** and is not ablated separately.
It is an implementation detail of the augmentation design, not a standalone claim:
global views double as the teacher's targets (`teacher_no_color_jitter` defaults to
false, so the teacher sees the same augmented crops), so heavy augmentation there
injects noise into the target the student must match, while local views only ever reach
the student.

## Configs

| File | Paper row | DAVC | ABM |
| --- | --- | --- | --- |
| `vitb_baseline.yaml` | Baseline | — | — |
| `vitb_plus_davc.yaml` | + DAVC | yes | — |
| `vitb_plus_abm.yaml` | + ABM | — | yes |
| `vitb_plus_davc_abm.yaml` | + DAVC + ABM (Full) | yes | yes |

The four form a complete 2x2. The cumulative chain
`Baseline -> +DAVC -> +DAVC+ABM` is a subset of it; `+ABM` alone is what lets you say
ABM contributes independently rather than only in the presence of DAVC. If compute is
tight, drop `vitb_plus_abm.yaml` first.

## Guarantees

All four configs are emitted by `make_configs.py` from one shared template. Everything
outside the `augmentation`, `masking`, `crops` and `train.output_dir` blocks is
byte-identical across all of them, and matches `xiaorong/vitb_plus_AB_C.yaml`: same
checkpoint, same 200 epochs, same batch size, LR schedule and seed. Edit the template in
`make_configs.py` and regenerate — do not hand-edit the YAML files.

```bash
python dinov3/configs/train/newdino/make_configs.py
```

## Verified

Each config was loaded through OmegaConf, merged with `ssl_default_config.yaml`, and
used to build a real `DataAugmentationDINO` and `MaskingGenerator`:

| config | strategy | local crop height / H | blur g1/g2/local | masking | mask density rows 8-11 |
| --- | --- | --- | --- | --- | --- |
| `vitb_baseline` | random | 0.17 – 0.17 | defaults | uniform | 0.356 |
| `vitb_plus_davc` | n_tcc | 0.28 – 0.53 | 0.30/0.10/0.40 | uniform | 0.359 |
| `vitb_plus_abm` | random | 0.17 – 0.17 | defaults | anatomy_biased | 0.431 |
| `vitb_plus_davc_abm` | n_tcc | 0.28 – 0.53 | 0.30/0.10/0.40 | anatomy_biased | 0.429 |

ABM raises masked-token density in the dental band rows (0.356 -> 0.431) as intended,
and n-TCC keeps local crops off the background that the legacy sampler reached into.
Outputs are finite in every case.

**Not verified:** no training run was started. This covers config parsing, augmentation
construction and a single forward pass through the transform only. Watch the first
iterations for loss sanity.

## Note on `local_crops_scale`

Baseline and `+ABM` use `[0.05, 0.32]`; DAVC configs use `[0.05, 0.26]`. These are not
the same knob. `random` parameterizes crop scale as a fraction of full image area via
`RandomResizedCrop`, which samples aspect ratio and therefore handles the ~2:1 panoramic
shape correctly. n-TCC parameterizes it as a fraction of `H^2`. Both are at their
respective natural settings; `0.32` is the official DINOv3 value and `0.26` is derived in
`toothdino_github_release_20260713/docs/CHANGES.md`.

## Running

```bash
python -m dinov3.train.train --config-file dinov3/configs/train/newdino/vitb_baseline.yaml
python -m dinov3.train.train --config-file dinov3/configs/train/newdino/vitb_plus_davc.yaml
python -m dinov3.train.train --config-file dinov3/configs/train/newdino/vitb_plus_abm.yaml
python -m dinov3.train.train --config-file dinov3/configs/train/newdino/vitb_plus_davc_abm.yaml
```

Afterwards, compare patch-feature PCA between the official DINOv3 checkpoint and the Full
run with `notebooks/pca.ipynb` for the qualitative figure.
