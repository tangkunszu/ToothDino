# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image

from .decoders import ImageDataDecoder
from .extended import ExtendedVisionDataset


class BCDataset(ExtendedVisionDataset):
    """
    Simple image-folder dataset that emits DINOv3-style samples.
    Each __getitem__ returns a list with a single dict containing
    global/local crops so that collate_data_and_cast can stack them.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
        )

        # Collect image files recursively so nested folders (e.g., Oral_RGB/) are included.
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = [
            str(p)
            for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix.lower() in exts
        ]

    # Longest side kept when decoding. The pipeline's largest consumer is a local crop of
    # side sqrt(local_crops_scale_min) * H resized to local_crops_size: with scale 0.05 and
    # crop size 128 that needs H >= 128 / sqrt(0.05) = 573, i.e. a ~1150 px long side on a
    # 2:1 panorama. 1600 leaves margin while capping the cost of the full-image medical
    # augmentation, which otherwise runs over 4.6M pixels on a 2976x1536 source. Set to
    # None to decode at native resolution.
    decode_max_side: Optional[int] = 1600

    def _open_capped(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        cap = self.decode_max_side
        if cap:
            w, h = image.size
            if max(w, h) > cap:
                # For JPEG this downscales inside the decoder (DCT domain), so the full
                # resolution is never materialised. No-op for PNG and friends.
                image.draft("RGB", (max(1, w * cap // max(w, h)), max(1, h * cap // max(w, h))))
                w, h = image.size
                if max(w, h) > cap:
                    # draft() is a no-op for PNG, so a real resize is still needed there.
                    # reducing_gap lets PIL do a cheap integer box-reduce first and only
                    # then a quality resample: 94.3 ms -> 82.3 ms of downstream augmentation
                    # cost per image in the pipeline benchmark, for the same output size.
                    scale = cap / max(w, h)
                    image = image.resize(
                        (max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.BICUBIC,
                        reducing_gap=2.0,
                    )
        return image.convert("RGB")

    def __getitem__(self, index: int):
        """
        Load an image, apply DataAugmentationDINO (passed via `transform`),
        attach bookkeeping fields, and wrap in a list expected by collate.
        """
        image_path = self.image_paths[index]
        try:
            image = self._open_capped(image_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        image_relpath = self.get_image_relpath(index)
        if self.transform is not None:
            if getattr(self.transform, "accepts_image_path", False):
                sample = self.transform(image, image_path=image_relpath)
            else:
                sample = self.transform(image)
        elif self.transforms is not None:
            # Fallback: a joint transform returning (image, target)
            image, _ = self.transforms(image, self.get_target(index))
            sample = {"global_crops": [image], "local_crops": []}
        else:
            sample = {"global_crops": [image], "local_crops": []}

        if not isinstance(sample, dict):
            raise TypeError(f"BCDataset transform must return a dict, got {type(sample)}")

        sample = dict(sample)
        sample["image_path"] = image_relpath
        sample["target"] = self.get_target(index)

        return [sample]

    def get_image_data(self, index: int) -> str:
        # Return the path so decoders can load lazily if needed.
        return self.image_paths[index]

    def get_image_relpath(self, index: int) -> str:
        return os.path.relpath(self.image_paths[index], self.root)

    def get_target(self, index: int) -> Any:
        # No labels are used during pretraining.
        return torch.zeros((1,))

    def __len__(self) -> int:
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = BCDataset(root=".")
    sample = dataset[0][0]
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample keys: {list(sample.keys())}")
