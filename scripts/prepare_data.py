from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRONE = ROOT / "data/raw/DroneVehicle/val"
DEFAULT_M3OT = ROOT / "data/raw/M3OT"
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
ALIASES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "freight car": "freight_car",
    "freight_car": "freight_car",
}


@dataclass(frozen=True)
class DronePair:
    sample_id: str
    scene: str
    rgb: Path
    thermal: Path
    rgb_xml: Path
    thermal_xml: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scene_from_xml(path: Path) -> str:
    root = ET.parse(path).getroot()
    return (root.findtext("folder") or "unknown").strip() or "unknown"


def discover_drone_pairs(root: Path) -> list[DronePair]:
    rgb_dir = root / "valimg"
    thermal_dir = root / "valimgr"
    rgb_labels = root / "vallabel"
    thermal_labels = root / "vallabelr"
    ids = sorted(path.stem for path in rgb_dir.glob("*.jpg"))
    rows: list[DronePair] = []
    for sample_id in ids:
        paths = (
            rgb_dir / f"{sample_id}.jpg",
            thermal_dir / f"{sample_id}.jpg",
            rgb_labels / f"{sample_id}.xml",
            thermal_labels / f"{sample_id}.xml",
        )
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Incomplete DroneVehicle pair {sample_id}: {paths}")
        rows.append(DronePair(sample_id, scene_from_xml(paths[2]), *paths))
    if len(rows) != 1469:
        raise RuntimeError(f"Expected 1469 local DroneVehicle pairs, found {len(rows)}")
    return rows


def assign_group_splits(rows: list[DronePair], seed: int) -> dict[str, str]:
    groups: dict[str, list[DronePair]] = defaultdict(list)
    for row in rows:
        groups[row.scene].append(row)
    if len(groups) < 3:
        raise RuntimeError(f"Need at least three scenes for a group-disjoint split, found {len(groups)}")

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    target = {"train": 0.70 * len(rows), "val": 0.15 * len(rows), "test": 0.15 * len(rows)}
    counts = Counter()
    assignment: dict[str, str] = {}
    for scene, members in group_items:
        split = max(target, key=lambda key: target[key] - counts[key])
        assignment[scene] = split
        counts[split] += len(members)
    if set(assignment.values()) != {"train", "val", "test"}:
        raise RuntimeError(f"Split assignment is incomplete: {counts}")
    return assignment


def parse_obb_labels(xml_path: Path, crop_top: int) -> tuple[list[str], Counter, int]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "840"))
    height = int(root.findtext("size/height", "712"))
    cropped_h = height - crop_top
    labels: list[str] = []
    class_counts: Counter = Counter()
    dropped = 0
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        name = ALIASES.get(raw_name)
        if name is None:
            dropped += 1
            continue
        polygon = obj.find("polygon")
        points: list[tuple[float, float]] = []
        if polygon is not None:
            for index in range(1, 5):
                x = polygon.findtext(f"x{index}")
                y = polygon.findtext(f"y{index}")
                if x is not None and y is not None:
                    points.append((float(x), float(y)))
        if len(points) != 4:
            box = obj.find("bndbox")
            if box is not None:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        if len(points) != 4:
            dropped += 1
            continue
        shifted = [(min(max(x, 0.0), width), min(max(y - crop_top, 0.0), cropped_h)) for x, y in points]
        xs = [point[0] for point in shifted]
        ys = [point[1] for point in shifted]
        if max(xs) - min(xs) < 1.0 or max(ys) - min(ys) < 1.0:
            dropped += 1
            continue
        values: list[float] = []
        for x, y in shifted:
            values.extend((x / width, y / cropped_h))
        cls = CLASSES.index(name)
        labels.append(str(cls) + " " + " ".join(f"{value:.8f}" for value in values))
        class_counts[name] += 1
    return labels, class_counts, dropped


def save_image(source: Path, target: Path, crop_top: int, thermal: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return
    with Image.open(source) as image:
        image = image.crop((0, crop_top, image.width, image.height))
        image = image.resize((640, 512), Image.Resampling.LANCZOS)
        if thermal:
            image = image.convert("L").convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(target, quality=95, subsampling=0)


def write_yolo_yaml(path: Path, dataset_root: Path) -> None:
    text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_scene_split_manifest(path: Path, pairs: list[DronePair]) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample = {row["sample_id"]: row for row in rows}
    if len(by_sample) != len(rows) or set(by_sample) != {pair.sample_id for pair in pairs}:
        raise ValueError("Split manifest must contain each discovered sample exactly once")
    assignment = {}
    counts, scenes = Counter(), defaultdict(set)
    for pair in pairs:
        row = by_sample[pair.sample_id]
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split}")
        if row.get("scene", pair.scene) != pair.scene:
            raise ValueError(f"Scene differs from source annotation for {pair.sample_id}")
        if pair.scene in assignment and assignment[pair.scene] != split:
            raise ValueError(f"Scene leakage for {pair.scene}")
        assignment[pair.scene] = split
        counts[split] += 1
        scenes[split].add(pair.scene)
    if dict(counts) != {"train": 1028, "val": 221, "test": 220}:
        raise ValueError(f"Manifest split counts differ from manuscript: {dict(counts)}")
    if {k: len(v) for k, v in scenes.items()} != {"train": 25, "val": 19, "test": 18}:
        raise ValueError("Manifest scene counts differ from manuscript 25/19/18")
    return assignment


def prepare_drone(root: Path, materialize: bool, seed: int, split_manifest: Path | None = None, new_split: bool = False) -> dict:
    pairs = discover_drone_pairs(root)
    if split_manifest is None and not new_split:
        raise ValueError("The original split manifest is required; use --new-split only for a separately labelled new experiment")
    scene_split = load_scene_split_manifest(split_manifest, pairs) if split_manifest else assign_group_splits(pairs, seed)
    manifest_path = ROOT / "manifests" / "dronevehicle_local_scene_disjoint_v1.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter = Counter()
    dropped = 0
    split_counts: Counter = Counter()
    scenes_by_split: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, str | int]] = []
    for index, pair in enumerate(pairs, start=1):
        split = scene_split[pair.scene]
        split_counts[split] += 1
        scenes_by_split[split].add(pair.scene)
        labels, counts, dropped_now = parse_obb_labels(pair.rgb_xml, crop_top=100)
        class_counts.update(counts)
        dropped += dropped_now
        rgb_hash = file_sha256(pair.rgb)
        thermal_hash = file_sha256(pair.thermal)
        rows.append({
            "sample_id": pair.sample_id,
            "scene": pair.scene,
            "split": split,
            "rgb_path": str(pair.rgb),
            "thermal_path": str(pair.thermal),
            "rgb_xml": str(pair.rgb_xml),
            "thermal_xml": str(pair.thermal_xml),
            "rgb_sha256": rgb_hash,
            "thermal_sha256": thermal_hash,
            "objects": len(labels),
        })
        if materialize:
            for modality, source, is_thermal in (
                ("rgb", pair.rgb, False),
                ("thermal", pair.thermal, True),
            ):
                dataset_root = ROOT / "data" / f"dronevehicle_{modality}"
                image_path = dataset_root / "images" / split / f"{pair.sample_id}.jpg"
                label_path = dataset_root / "labels" / split / f"{pair.sample_id}.txt"
                save_image(source, image_path, crop_top=100, thermal=is_thermal)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        if index % 250 == 0:
            print(f"DroneVehicle: processed {index}/{len(pairs)}")

    if not rows:
        raise ValueError(f"No paired samples found under {root}")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if materialize:
        for modality in ("rgb", "thermal"):
            dataset_root = ROOT / "data" / f"dronevehicle_{modality}"
            write_yolo_yaml(ROOT / "configs" / "data" / f"dronevehicle_{modality}.yaml", dataset_root)
        paired = ROOT / "configs/data/dronevehicle_paired.yaml"
        write_yolo_yaml(paired, ROOT / "data/dronevehicle_rgb")
        with paired.open("a", encoding="utf-8") as handle:
            handle.write("channels: 4\n" + f"thermal_root: {(ROOT / 'data/dronevehicle_thermal').as_posix()}\n")
    scene_sets = [scenes_by_split[key] for key in ("train", "val", "test")]
    if any(scene_sets[i] & scene_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Scene leakage detected")
    return {
        "split_origin": "provided_manifest" if split_manifest else "new_experiment_not_paper_split",
        "pairs": len(pairs),
        "split_counts": dict(split_counts),
        "scene_counts": {key: len(value) for key, value in scenes_by_split.items()},
        "class_counts": dict(class_counts),
        "dropped_objects": dropped,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def prepare_m3ot(root: Path) -> dict:
    manifest_path = ROOT / "manifests" / "m3ot_pairs.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    missing: list[str] = []
    split_counts: Counter = Counter()
    for group in ("1", "2"):
        for split in ("train", "val", "test"):
            rgb_split = root / group / "rgb" / split
            ir_split = root / group / "ir" / split
            for rgb_sequence in sorted(path for path in rgb_split.iterdir() if path.is_dir()):
                ir_sequence = ir_split / f"{rgb_sequence.name}T"
                rgb_image_dir = rgb_sequence / "img1"
                ir_image_dir = ir_sequence / "img1"
                if not rgb_image_dir.is_dir() or not ir_image_dir.is_dir():
                    raise FileNotFoundError(f"Missing M3OT img1 directory for {rgb_sequence}")
                rgb_files = {path.stem: path for path in rgb_image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
                ir_files = {path.stem: path for path in ir_image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
                shared = sorted(set(rgb_files) & set(ir_files))
                for stem in shared:
                    rows.append({
                        "group": group,
                        "split": split,
                        "sequence": rgb_sequence.name,
                        "frame": stem,
                        "rgb_path": str(rgb_files[stem]),
                        "thermal_path": str(ir_files[stem]),
                        "rgb_size": rgb_files[stem].stat().st_size,
                        "thermal_size": ir_files[stem].stat().st_size,
                    })
                    split_counts[split] += 1
                for stem in sorted(set(rgb_files) ^ set(ir_files)):
                    missing.append(f"{group}/{split}/{rgb_sequence.name}/{stem}")
    if not rows:
        raise ValueError(f"No paired samples found under {root}")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = {"train": 8630, "val": 1200, "test": 960}
    if dict(split_counts) != expected:
        raise RuntimeError(f"M3OT counts changed: expected {expected}, found {dict(split_counts)}")
    return {
        "pairs": len(rows),
        "split_counts": dict(split_counts),
        "unmatched": missing,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone-root", type=Path, default=DEFAULT_DRONE)
    parser.add_argument("--m3ot-root", type=Path, default=DEFAULT_M3OT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--new-split", action="store_true", help="Create a new scene split; does not reproduce manuscript membership")
    parser.add_argument("--skip-m3ot", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--clean-derived", action="store_true")
    args = parser.parse_args()
    if args.clean_derived:
        # Only generated modality folders, never data/raw or arbitrary directories.
        for modality in ("rgb", "thermal"):
            derived = ROOT / "data" / f"dronevehicle_{modality}"
            if derived.is_dir() and not derived.is_symlink():
                shutil.rmtree(derived)
    summary = {
        "protocol": "local_scene_disjoint_v1",
        "seed": args.seed,
        "crop_top": 100,
        "output_size": [640, 512],
        "dronevehicle": prepare_drone(args.drone_root, args.materialize, args.seed, args.split_manifest, args.new_split),
        "m3ot": None if args.skip_m3ot else prepare_m3ot(args.m3ot_root),
    }
    target = ROOT / "artifacts" / "data_qa_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
