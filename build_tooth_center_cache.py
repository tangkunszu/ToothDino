"""Precompute the n-TCC representative tooth centers for every training image.

The dental band estimate is a deterministic function of the image, but it is recomputed
on every epoch for every sample (~11 ms per image, 4% of the augmentation cost). This
writes it once to a JSON cache that DataAugmentationDINO reads via
crops.representative_tooth_cache_path.

Keys are paths relative to the dataset root, matching BCDataset.get_image_relpath.

Caveat: centers are computed on the undistorted image, while at training time the band
would otherwise be estimated after MedicalImageAugmentation (which applies a 0-5 degree
rotation with p=0.3). Cached centers therefore ignore that rotation. The band is a coarse
prior stabilised by _stabilize_panorama_focus_band, so this is well inside its own noise,
but it is a real difference from the uncached path.

Usage:
    python build_tooth_center_cache.py --root /data/tangkun/pXray/All/train/pXray \
        --out /data/tangkun/pXray/All/train/tooth_center_cache.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, "/data/tangkun/project/dinov3newimprove/dinov3")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_AUG = None


def _get_aug(local_crops_number, global_crops_size, local_crops_size):
    global _AUG
    if _AUG is None:
        from dinov3.data.augmentations import DataAugmentationDINO

        _AUG = DataAugmentationDINO(
            global_crops_scale=(0.32, 1.0), local_crops_scale=(0.05, 0.26),
            local_crops_number=local_crops_number, global_crops_size=global_crops_size,
            local_crops_size=local_crops_size, augmentation_mode="dental_xray",
            local_crop_strategy="n_tcc", medical_augmentation=False, horizontal_flips=True,
        )
    return _AUG


def _one(args):
    path, relpath, ln, gs, ls = args
    try:
        with Image.open(path) as im:
            centers, _ = _get_aug(ln, gs, ls)._get_representative_tooth_centers_fast(im.convert("RGB"))
        return relpath, [[round(float(x), 5), round(float(y), 5)] for x, y in centers]
    except Exception as exc:  # a broken file should not kill the whole build
        return relpath, {"error": str(exc)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--local-crops-number", type=int, default=8)
    ap.add_argument("--global-crops-size", type=int, default=256)
    ap.add_argument("--local-crops-size", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    print(f"{len(paths)} images under {root}")

    jobs = [(str(p), os.path.relpath(str(p), str(root)),
             args.local_crops_number, args.global_crops_size, args.local_crops_size)
            for p in paths]

    entries, failed, t0 = {}, 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (rel, centers) in enumerate(ex.map(_one, jobs, chunksize=64), 1):
            if isinstance(centers, dict):
                failed += 1
            else:
                entries[rel] = {"centers": centers}
            if i % 5000 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s  eta {el/i*(len(jobs)-i):.0f}s")

    payload = {
        "version": 1,
        "root": str(root),
        "num_entries": len(entries),
        "local_crops_number": args.local_crops_number,
        "note": "centers computed without MedicalImageAugmentation; see module docstring",
        "entries": entries,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {len(entries)} entries ({failed} failed) to {args.out}  [{size_mb:.1f} MB]"
          f"  in {time.perf_counter()-t0:.0f}s")
    print("\nEnable with, in the crops block of the training config:")
    print(f"  representative_tooth_cache_path: {args.out}")


if __name__ == "__main__":
    main()
