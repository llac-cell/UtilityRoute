"""M3OT paired-frame manifest and MOT/COCO annotation helpers."""

from __future__ import annotations

import configparser
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class M3OTFrame:
    group: str
    split: str
    sequence: str
    frame: int
    rgb_path: Path
    thermal_path: Path
    rgb_size: int
    thermal_size: int

    @property
    def sequence_dir(self) -> Path:
        return self.rgb_path.parent.parent


@dataclass(frozen=True)
class MOTObject:
    track_id: int
    box: np.ndarray
    confidence: float
    class_id: int
    visibility: float


def platform_path(value: str | Path) -> Path:
    """Resolve manifest WSL paths when the audit is invoked from Windows."""

    path = Path(value)
    if path.exists() or os.name != "nt":
        return path
    text = str(value).replace("\\", "/")
    parts = text.split("/")
    if len(parts) > 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        return Path(f"{parts[2].upper()}:\\", *parts[3:])
    return path


def load_m3ot_manifest(manifest: Path, split: str) -> list[M3OTFrame]:
    """Read one deterministic split from the checked-in pair manifest."""

    frames: list[M3OTFrame] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            frames.append(
                M3OTFrame(
                    group=row["group"],
                    split=row["split"],
                    sequence=row["sequence"],
                    frame=int(row["frame"]),
                    rgb_path=platform_path(row["rgb_path"]),
                    thermal_path=platform_path(row["thermal_path"]),
                    rgb_size=int(row["rgb_size"]),
                    thermal_size=int(row["thermal_size"]),
                )
            )
    frames.sort(key=lambda item: (item.group, item.sequence, item.frame))
    keys = [(item.group, item.sequence, item.frame) for item in frames]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate M3OT manifest keys in split {split}")
    if not frames:
        raise ValueError(f"No M3OT frames found for split {split}: {manifest}")
    return frames


def load_mot_ground_truth(sequence_dir: Path) -> dict[int, list[MOTObject]]:
    """Load MOTChallenge ``gt.txt`` rows, retaining the source floating boxes."""

    by_frame: dict[int, list[MOTObject]] = defaultdict(list)
    gt_path = sequence_dir / "gt" / "gt.txt"
    with gt_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            values = line.strip().split(",")
            if len(values) < 9:
                raise ValueError(f"Malformed MOT row {gt_path}:{line_number}")
            frame, track_id = int(values[0]), int(values[1])
            x, y, width, height = (float(value) for value in values[2:6])
            by_frame[frame].append(
                MOTObject(
                    track_id=track_id,
                    box=np.asarray([x, y, x + width, y + height], dtype=np.float64),
                    confidence=float(values[6]),
                    class_id=int(float(values[7])),
                    visibility=float(values[8]),
                )
            )
    return dict(by_frame)


def sequence_info(sequence_dir: Path) -> dict[str, int | str]:
    parser = configparser.ConfigParser()
    path = sequence_dir / "seqinfo.ini"
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)
    section = parser["Sequence"]
    return {
        "name": section["name"],
        "frame_rate": int(section["frameRate"]),
        "sequence_length": int(section["seqLength"]),
        "image_width": int(section["imWidth"]),
        "image_height": int(section["imHeight"]),
        "image_extension": section["imExt"],
    }


def group_sequences(frames: Iterable[M3OTFrame]) -> dict[tuple[str, str], list[M3OTFrame]]:
    grouped: dict[tuple[str, str], list[M3OTFrame]] = defaultdict(list)
    for frame in frames:
        grouped[(frame.group, frame.sequence)].append(frame)
    return dict(grouped)


def audit_m3ot_split(frames: list[M3OTFrame], m3ot_root: Path) -> dict[str, object]:
    """Cross-check manifest, seqinfo, native MOT GT, and RGB COCO exports."""

    grouped = group_sequences(frames)
    missing_files: list[str] = []
    size_mismatches: list[str] = []
    checked_files = 0
    sequence_errors: list[str] = []
    mot_rows: dict[tuple[str, str, int, int], np.ndarray] = {}
    mot_gt_rows_raw = 0
    mot_gt_rows_filtered_out = 0
    sequence_rows: list[dict[str, object]] = []
    for (group, sequence), sequence_frames in sorted(grouped.items()):
        # The full evaluator opens every selected image. The annotation audit
        # checks sequence endpoints here to avoid thousands of slow metadata
        # calls on a mounted Windows volume during every smoke run.
        endpoint_frames = sequence_frames if len(sequence_frames) == 1 else [sequence_frames[0], sequence_frames[-1]]
        for frame in endpoint_frames:
            for path, expected_size in ((frame.rgb_path, frame.rgb_size), (frame.thermal_path, frame.thermal_size)):
                checked_files += 1
                if not path.is_file():
                    missing_files.append(str(path))
                elif path.stat().st_size != expected_size:
                    size_mismatches.append(str(path))
        info = sequence_info(sequence_frames[0].sequence_dir)
        manifest_numbers = [item.frame for item in sequence_frames]
        expected_numbers = list(range(1, int(info["sequence_length"]) + 1))
        if manifest_numbers != expected_numbers:
            sequence_errors.append(f"{group}/{sequence}: manifest frame range does not match seqinfo")
        if info["name"] != sequence:
            sequence_errors.append(f"{group}/{sequence}: seqinfo name={info['name']}")
        gt = load_mot_ground_truth(sequence_frames[0].sequence_dir)
        raw_gt_count = sum(len(objects) for objects in gt.values())
        mot_gt_rows_raw += raw_gt_count
        gt_count = 0
        track_ids: set[int] = set()
        for frame_number, objects in gt.items():
            for obj in objects:
                if obj.confidence <= 0 or obj.class_id != 1:
                    mot_gt_rows_filtered_out += 1
                    continue
                mot_rows[(group, sequence, frame_number, obj.track_id)] = obj.box
                track_ids.add(obj.track_id)
                gt_count += 1
        sequence_rows.append(
            {
                "group": group,
                "sequence": sequence,
                "manifest_frames": len(sequence_frames),
                "seqinfo_frames": int(info["sequence_length"]),
                "frame_rate": int(info["frame_rate"]),
                "mot_gt_rows_raw": raw_gt_count,
                "mot_gt_rows": gt_count,
                "mot_gt_rows_filtered_out": raw_gt_count - gt_count,
                "mot_unique_track_ids": len(track_ids),
                "empty_gt_frames": int(info["sequence_length"]) - len(gt),
            }
        )

    split = frames[0].split
    manifest_keys = {(item.group, item.sequence, item.frame) for item in frames}
    coco_keys: set[tuple[str, str, int]] = set()
    coco_rows: dict[tuple[str, str, int, int], np.ndarray] = {}
    coco_frame_index_errors = 0
    coco_dimension_errors = 0
    coco_annotation_count = 0
    for group in sorted({item.group for item in frames}):
        annotation_path = m3ot_root / "Annotations" / group / "rgb" / f"{split}_cocoformat.json"
        with annotation_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        image_keys: dict[int, tuple[str, str, int]] = {}
        for image in data["images"]:
            sequence = Path(image["file_name"]).parts[-3]
            frame_number = int(image["mot_frame_id"])
            key = (group, sequence, frame_number)
            coco_keys.add(key)
            image_keys[int(image["id"])] = key
            if int(image["frame_id"]) != frame_number - 1:
                coco_frame_index_errors += 1
            if int(image["width"]) != 640 or int(image["height"]) != 512:
                coco_dimension_errors += 1
        for annotation in data["annotations"]:
            key = image_keys[int(annotation["image_id"])]
            track_id = int(annotation["mot_instance_id"])
            x, y, width, height = (float(value) for value in annotation["bbox"])
            coco_rows[(*key, track_id)] = np.asarray([x, y, x + width, y + height], dtype=np.float64)
            coco_annotation_count += 1

    shared_annotations = sorted(mot_rows.keys() & coco_rows.keys())
    bbox_delta = [float(np.max(np.abs(mot_rows[key] - coco_rows[key]))) for key in shared_annotations]
    annotation_key_missing = sorted(mot_rows.keys() - coco_rows.keys())
    annotation_key_extra = sorted(coco_rows.keys() - mot_rows.keys())
    missing_coco_frames = sorted(manifest_keys - coco_keys)
    extra_coco_frames = sorted(coco_keys - manifest_keys)
    issues = (
        len(missing_files)
        + len(size_mismatches)
        + len(sequence_errors)
        + len(missing_coco_frames)
        + len(extra_coco_frames)
        + len(annotation_key_missing)
        + len(annotation_key_extra)
        + coco_frame_index_errors
        + coco_dimension_errors
    )
    return {
        "status": "pass" if issues == 0 else "warning",
        "split": split,
        "manifest_frames": len(frames),
        "sequences": len(grouped),
        "missing_files": len(missing_files),
        "endpoint_files_checked": checked_files,
        "file_size_mismatches": len(size_mismatches),
        "sequence_errors": sequence_errors,
        "missing_coco_frames": len(missing_coco_frames),
        "extra_coco_frames": len(extra_coco_frames),
        "extra_coco_frame_examples": ["/".join(map(str, key)) for key in extra_coco_frames[:10]],
        "coco_frame_index_errors": coco_frame_index_errors,
        "coco_dimension_errors": coco_dimension_errors,
        "annotation_modality": "rgb",
        "mot_gt_rows_raw": mot_gt_rows_raw,
        "mot_gt_rows": len(mot_rows),
        "mot_gt_rows_filtered_out": mot_gt_rows_filtered_out,
        "coco_annotation_rows": coco_annotation_count,
        "annotation_keys_missing_from_coco": len(annotation_key_missing),
        "annotation_keys_extra_in_coco": len(annotation_key_extra),
        "shared_annotation_rows": len(shared_annotations),
        "coco_bbox_exact_rows": int(sum(delta == 0.0 for delta in bbox_delta)),
        "coco_bbox_max_abs_delta_pixels": max(bbox_delta, default=0.0),
        "bbox_authority": "native MOT gt/gt.txt floating-point xywh",
        "sequence_details": sequence_rows,
    }
