"""Convert sharded eval teacher checkpoints to the single-file .pth layout.

Runs launched with the base conda env (torch 2.7.1) hit the guard at train.py:373 --
torch 2.7's FSDP keeps the EMA parameters as DTensor, and `has_dtensor and "nccl" in
backend` makes the trainer fall back to `sharded_teacher_checkpoint/*.distcp` instead of
writing `teacher_checkpoint.pth`. The dinov3 conda env (torch 2.5.0) does not trigger it.

This rebuilds the single-file layout after the fact. Verified against
xiaorong_vitb/plus_ABC_toothaware_direct_new/eval/training_44599/teacher_checkpoint.pth:
202 parameters, identical key set, no shape or dtype mismatch. Only the outer wrapper
differs -- dcp_to_torch_save emits {'iteration', 'model'} where the trainer writes
{'teacher'} -- so the wrapper is rewritten here.

Usage:
    python convert_sharded_checkpoints.py output/newdino/vitb_plus_davc_abm
    python convert_sharded_checkpoints.py <run_dir> --keep-intermediate
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def convert_one(sharded_dir: Path, out_path: Path, keep_intermediate: bool = False) -> int:
    tmp = Path(tempfile.mkdtemp()) / "raw.pth"
    dcp_to_torch_save(str(sharded_dir), str(tmp))
    raw = torch.load(tmp, map_location="cpu", weights_only=False)

    if "model" in raw:
        state = raw["model"]
    elif "teacher" in raw:
        state = raw["teacher"]
    else:
        raise RuntimeError(f"unexpected top-level keys in {sharded_dir}: {list(raw)[:5]}")

    torch.save({"teacher": state}, out_path)
    n = len(state)
    if not keep_intermediate:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return n


def extract_backbone(pth_path: Path, out_path: Path) -> int:
    """Strip the 'backbone.' prefix, the layout downstream tasks load (cf. rename.py)."""
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)["teacher"]
    backbone = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not backbone:
        raise RuntimeError(f"no backbone.* keys in {pth_path}")
    torch.save(backbone, out_path)
    return len(backbone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="training output dir containing eval/training_*/")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extract-backbone", action="store_true",
                    help="also write backbone_only.pth next to each teacher_checkpoint.pth")
    ap.add_argument("--backbone-dir", default=None,
                    help="collect the backbone-only files into this directory instead")
    args = ap.parse_args()

    eval_root = Path(args.run_dir) / "eval"
    if not eval_root.is_dir():
        raise SystemExit(f"no eval/ under {args.run_dir}")

    found = sorted(eval_root.glob("training_*/sharded_teacher_checkpoint"))
    if not found:
        print(f"no sharded checkpoints under {eval_root} -- nothing to do")
        return

    bb_dir = Path(args.backbone_dir) if args.backbone_dir else None
    if bb_dir:
        bb_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(found)} sharded checkpoint(s) under {eval_root}\n")
    for sharded in found:
        tag = sharded.parent.name
        out = sharded.parent / "teacher_checkpoint.pth"
        if out.exists() and not args.overwrite:
            print(f"  {tag}: teacher_checkpoint.pth already exists, skipping convert")
        else:
            n = convert_one(sharded, out, args.keep_intermediate)
            print(f"  {tag}: {n} params -> {out.name} [{os.path.getsize(out)/1e6:.0f} MB]")

        if args.extract_backbone or bb_dir:
            bb_out = (bb_dir / f"{Path(args.run_dir).name}_{tag}_backbone.pth") if bb_dir \
                else sharded.parent / "backbone_only.pth"
            if bb_out.exists() and not args.overwrite:
                print(f"  {tag}: {bb_out.name} already exists, skipping extract")
            else:
                m = extract_backbone(out, bb_out)
                print(f"  {tag}: backbone {m} keys -> {bb_out} "
                      f"[{os.path.getsize(bb_out)/1e6:.0f} MB]")

    print("\nLoad exactly like the checkpoints from a torch-2.5 run:")
    print("  sd = torch.load(path, map_location='cpu')['teacher']")
    print("  # keys are backbone.* ; strip the 'backbone.' prefix for a bare ViT")


if __name__ == "__main__":
    main()
