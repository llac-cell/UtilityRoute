from __future__ import annotations

import math
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset


class PairedYOLODataset(YOLODataset):
    """Paired RGB + thermal BCHW data with shared spatial transforms and RGB OBB labels."""

    def build_transforms(self, hyp=None):
        # The upstream defaults add unreported blur/CLAHE via Albumentations.
        # These may also reject four-channel images. Keep only manuscript transforms.
        transforms = super().build_transforms(hyp)
        transforms.transforms = [t for t in transforms.transforms if type(t).__name__ != "Albumentations"]
        return transforms

    def paired_thermal_path(self, rgb_path: str) -> Path:
        thermal_root = self.data.get("thermal_root")
        if thermal_root:
            rgb_root = Path(self.data["path"]).expanduser().resolve()
            relative = Path(rgb_path).resolve().relative_to(rgb_root)
            return Path(thermal_root).expanduser().resolve() / relative
        return self.thermal_path(rgb_path)

    @staticmethod
    def thermal_path(rgb_path: str) -> Path:
        path = Path(rgb_path)
        parts = list(path.parts)
        try:
            index = parts.index("dronevehicle_rgb")
        except ValueError as exc:
            raise ValueError(f"RGB path does not contain dronevehicle_rgb: {path}") from exc
        parts[index] = "dronevehicle_thermal"
        return Path(*parts)

    def load_image(self, i: int, rect_mode: bool = True):
        im = self.ims[i]
        if im is None:
            rgb_path = self.im_files[i]
            thermal_path = self.paired_thermal_path(rgb_path)
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            thermal = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
            if rgb is None or thermal is None:
                raise FileNotFoundError(f"Missing paired image: {rgb_path} / {thermal_path}")
            if thermal.shape != rgb.shape[:2]:
                raise ValueError(f"Paired image shapes differ: {rgb_path} {rgb.shape[:2]} / {thermal_path} {thermal.shape}")
            # Upstream Format reverses BGR only for three-channel data. Convert here
            # so that the four-channel output is R,G,B,T rather than B,G,R,T.
            im = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), thermal[..., None]], axis=2)
            h0, w0 = im.shape[:2]
            if rect_mode:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    w = min(math.ceil(w0 * ratio), self.imgsz)
                    h = min(math.ceil(h0 * ratio), self.imgsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    old = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[old], self.im_hw0[old], self.im_hw[old] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]


def read_utility_targets(path: str | Path, expected_checkpoint_sha256: str | None = None) -> dict[tuple[str, str], torch.Tensor]:
    targets = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split, sample_id = row.get("split", ""), row.get("sample_id", "")
            if split not in {"train", "val"} or not sample_id:
                raise ValueError("Training utility cache requires explicit train/val split and sample_id; test rows are forbidden")
            key = (split, sample_id)
            if expected_checkpoint_sha256:
                if row.get("checkpoint_sha256") != expected_checkpoint_sha256:
                    raise ValueError(f"Utility target checkpoint differs from frozen expert initialization: {key}")
                if row.get("preprocessing") != "rgbt_square640_no_augmentation_v2":
                    raise ValueError(f"Utility cache preprocessing is missing or incompatible: {key}")
            if key in targets:
                raise ValueError(f"Duplicate utility cache sample: {key}")
            values = torch.tensor([float(row[f"utility_{i}"]) for i in range(4)], dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite utility target: {key}")
            if "identity_loss" in row:
                identity = float(row["identity_loss"])
                expected = torch.tensor([(identity - float(row[f"expert_{i}_loss"])) / (identity + 1e-6) for i in range(4)])
                if identity < 0 or not torch.allclose(values, expected, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Utility formula mismatch: {key}")
            targets[key] = values
    if not targets:
        raise ValueError("Utility cache is empty")
    return targets


class UtilityPairedYOLODataset(PairedYOLODataset):
    """Offline targets describe unaugmented individual square-letterboxed frames."""
    def __init__(self, *args, utility_cache: str | Path, utility_split: str | None = None,
                 expected_checkpoint_sha256: str | None = None, **kwargs):
        if kwargs.get("augment", False):
            raise ValueError("Offline frame utility targets cannot supervise mosaic/geometric augmentations; use online teacher targets")
        if kwargs.get("rect", False):
            raise ValueError("Offline cache uses square letterboxing (rect=False)")
        if expected_checkpoint_sha256 and kwargs.get("imgsz", 640) != 640:
            raise ValueError("The offline cache was generated at square640; router imgsz must be 640")
        self.utility_targets = read_utility_targets(utility_cache, expected_checkpoint_sha256)
        self.utility_split = utility_split
        super().__init__(*args, **kwargs)
        for name in self.im_files:
            self._target_key(name)

    def _target_key(self, filename):
        sample_id = Path(filename).stem
        split = self.utility_split
        if split is None:
            candidates = [key for key in self.utility_targets if key[1] == sample_id]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing split for utility sample {sample_id}")
            return candidates[0]
        key = (split, sample_id)
        if key not in self.utility_targets:
            raise KeyError(f"No utility target for {key}")
        return key

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        item["utility_target"] = self.utility_targets[self._target_key(item["im_file"])]
        return item

    @staticmethod
    def collate_fn(batch):
        # Do not mutate dataset samples while collating.
        copied = [dict(item) for item in batch]
        targets = [item.pop("utility_target") for item in copied]
        output = YOLODataset.collate_fn(copied)
        output["utility_target"] = torch.stack(targets)
        return output
