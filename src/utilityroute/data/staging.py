from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


SPLITS = ("train", "val", "test")
MODALITIES = ("rgb", "thermal")
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}
UNSAFE_STAGING_FILESYSTEMS = {"9p", "drvfs", "fuseblk"}


@dataclass(frozen=True)
class DatasetFile:
    sample_id: str
    split: str
    modality: str
    kind: str
    source: Path
    relative_path: Path


@dataclass(frozen=True)
class InventoryRow:
    sample_id: str
    split: str
    modality: str
    kind: str
    relative_path: str
    source_path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_type(path: Path, mountinfo: Path = Path("/proc/self/mountinfo")) -> str:
    """Return the filesystem type for path using Linux mountinfo."""
    if not mountinfo.is_file():
        return "unknown"
    probe = path.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    resolved = probe.resolve()
    matches: list[tuple[int, str]] = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        rhs = right.split()
        if len(fields) < 5 or not rhs:
            continue
        mount_point = Path(_decode_mount_path(fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((len(str(mount_point)), rhs[0].lower()))
    if not matches:
        return "unknown"
    return max(matches)[1]


def assert_safe_destination_filesystem(destination: Path) -> str:
    fs_type = filesystem_type(destination)
    if fs_type in UNSAFE_STAGING_FILESYSTEMS:
        raise ValueError(
            f"destination {destination} is on {fs_type}, not WSL ext4; "
            "choose a path under /home (for example /home/kun/utilityroute_staging)"
        )
    if fs_type != "ext4":
        raise ValueError(f"destination {destination} must be on ext4, detected {fs_type!r}")
    return fs_type


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)} and at least one row")
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        split = row["split"]
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        if split not in SPLITS:
            raise ValueError(f"unsupported split {split!r} for sample {sample_id}")
        seen.add(sample_id)
    return rows


def expected_dataset_files(source_root: Path, manifest_rows: Iterable[dict[str, str]]) -> list[DatasetFile]:
    files: list[DatasetFile] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        split = row["split"]
        for modality in MODALITIES:
            dataset_name = f"dronevehicle_{modality}"
            image_rel = Path(dataset_name, "images", split, f"{sample_id}.jpg")
            label_rel = Path(dataset_name, "labels", split, f"{sample_id}.txt")
            files.extend(
                (
                    DatasetFile(sample_id, split, modality, "image", source_root / image_rel, image_rel),
                    DatasetFile(sample_id, split, modality, "label", source_root / label_rel, label_rel),
                )
            )
    return files


def _stat_with_retry(path: Path, retries: int) -> os.stat_result:
    delay = 0.1
    for attempt in range(retries + 1):
        try:
            return path.stat()
        except FileNotFoundError:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    raise AssertionError("unreachable")


def validate_source_tree(files: list[DatasetFile], retries: int = 5) -> dict[str, object]:
    missing: list[str] = []
    expected_by_dir: dict[Path, set[str]] = {}
    for item in files:
        expected_by_dir.setdefault(item.source.parent, set()).add(item.source.name)
        try:
            _stat_with_retry(item.source, retries)
        except FileNotFoundError:
            missing.append(str(item.source))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} prepared dataset files are missing:\n{preview}")

    extras: dict[str, list[str]] = {}
    for directory, expected_names in expected_by_dir.items():
        suffix = ".jpg" if "images" in directory.parts else ".txt"
        actual_names = {path.name for path in directory.glob(f"*{suffix}")}
        extra = sorted(actual_names - expected_names)
        absent = sorted(expected_names - actual_names)
        if absent:
            raise FileNotFoundError(f"{directory} is missing {len(absent)} expected files: {absent[:20]}")
        if extra:
            extras[str(directory)] = extra
    split_counts = {
        split: sum(1 for item in files if item.split == split and item.kind == "image") // len(MODALITIES)
        for split in SPLITS
    }
    return {"dataset_files": len(files), "samples_by_split": split_counts, "ignored_extras": extras}


def _copy_verified(item: DatasetFile, partial_root: Path, retries: int) -> InventoryRow:
    destination = partial_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.1
    for attempt in range(retries + 1):
        try:
            before = item.source.stat()
            digest = hashlib.sha256()
            with item.source.open("rb") as source, destination.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                target.flush()
                os.fsync(target.fileno())
            after = item.source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"source changed while staging: {item.source}")
            source_hash = digest.hexdigest()
            staged_hash = sha256_file(destination)
            if source_hash != staged_hash or after.st_size != destination.stat().st_size:
                raise OSError(f"copy verification failed for {item.source}")
            return InventoryRow(
                sample_id=item.sample_id,
                split=item.split,
                modality=item.modality,
                kind=item.kind,
                relative_path=item.relative_path.as_posix(),
                source_path=str(item.source),
                size_bytes=after.st_size,
                sha256=source_hash,
            )
        except FileNotFoundError:
            if destination.exists():
                destination.unlink()
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    raise AssertionError("unreachable")


def inventory_sha256(rows: Iterable[InventoryRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value.relative_path):
        digest.update(f"{row.relative_path}\t{row.size_bytes}\t{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[InventoryRow]) -> None:
    fieldnames = list(asdict(rows[0]))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def staged_data_configs(destination: Path) -> dict[str, dict[str, object]]:
    common = {"train": "images/train", "val": "images/val", "test": "images/test", "names": NAMES}
    return {
        "dronevehicle_rgb.yaml": {"path": str(destination / "dronevehicle_rgb"), **common},
        "dronevehicle_thermal.yaml": {"path": str(destination / "dronevehicle_thermal"), **common},
        "dronevehicle_paired.yaml": {
            "path": str(destination / "dronevehicle_rgb"),
            **common,
            "channels": 4,
        },
    }


def stage_dataset(
    source_root: Path,
    manifest_path: Path,
    destination: Path,
    *,
    retries: int = 5,
    require_ext4: bool = False,
) -> dict[str, object]:
    source_root = source_root.resolve()
    manifest_path = manifest_path.resolve()
    destination = destination.expanduser().resolve()
    if require_ext4:
        destination_fs = assert_safe_destination_filesystem(destination)
    else:
        destination_fs = filesystem_type(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")

    manifest_rows = read_manifest_rows(manifest_path)
    files = expected_dataset_files(source_root, manifest_rows)
    source_validation = validate_source_tree(files, retries=retries)
    partial = destination.with_name(f"{destination.name}.partial-{os.getpid()}-{time.time_ns()}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        inventory = [_copy_verified(item, partial, retries) for item in files]
        provenance_dir = partial / "provenance"
        config_dir = partial / "configs"
        provenance_dir.mkdir()
        config_dir.mkdir()
        _write_csv(provenance_dir / "inventory.csv", inventory)
        for name, config in staged_data_configs(destination).items():
            with (config_dir / name).open("x", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        report: dict[str, object] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_filesystem": filesystem_type(source_root),
            "destination": str(destination),
            "destination_filesystem": destination_fs,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_semantics": (
                "The manifest defines sample membership and split only. Its rgb_sha256/thermal_sha256 "
                "refer to incoming raw images; prepared images were cropped/resized and are verified by inventory.csv."
            ),
            "copy_policy": "copy-only, sequential, retry transient FileNotFound, SHA-256 verify destination",
            "source_validation": source_validation,
            "inventory_rows": len(inventory),
            "inventory_sha256": inventory_sha256(inventory),
            "data_yamls": {name: str(destination / "configs" / name) for name in staged_data_configs(destination)},
        }
        with (provenance_dir / "stage_report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        partial.rename(destination)
        return report
    except Exception:
        # Preserve the partial directory for forensic inspection; never delete source or an existing stage.
        raise


def verify_staged_dataset(destination: Path) -> dict[str, object]:
    destination = destination.expanduser().resolve()
    inventory_path = destination / "provenance" / "inventory.csv"
    report_path = destination / "provenance" / "stage_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[InventoryRow] = []
    errors: list[str] = []
    for raw in raw_rows:
        row = InventoryRow(
            sample_id=raw["sample_id"],
            split=raw["split"],
            modality=raw["modality"],
            kind=raw["kind"],
            relative_path=raw["relative_path"],
            source_path=raw["source_path"],
            size_bytes=int(raw["size_bytes"]),
            sha256=raw["sha256"],
        )
        path = destination / row.relative_path
        if not path.is_file():
            errors.append(f"missing: {row.relative_path}")
        elif path.stat().st_size != row.size_bytes:
            errors.append(f"size mismatch: {row.relative_path}")
        elif sha256_file(path) != row.sha256:
            errors.append(f"sha256 mismatch: {row.relative_path}")
        rows.append(row)
    actual_inventory_hash = inventory_sha256(rows)
    if actual_inventory_hash != report["inventory_sha256"]:
        errors.append("inventory digest does not match stage_report.json")
    if errors:
        raise RuntimeError("staged dataset verification failed:\n" + "\n".join(errors[:20]))
    return {
        "destination": str(destination),
        "filesystem": filesystem_type(destination),
        "verified_files": len(rows),
        "inventory_sha256": actual_inventory_hash,
        "ok": True,
    }


def dry_run_plan(source_root: Path, manifest_path: Path, destination: Path, retries: int = 5,
                 *, require_ext4: bool = False) -> dict[str, object]:
    rows = read_manifest_rows(manifest_path)
    files = expected_dataset_files(source_root, rows)
    source_validation = validate_source_tree(files, retries=retries)
    return {
        "action": "dry_run",
        "source_root": str(source_root.resolve()),
        "source_filesystem": filesystem_type(source_root),
        "destination": str(destination.expanduser().resolve()),
        "destination_filesystem": (assert_safe_destination_filesystem(destination)
                                   if require_ext4 else filesystem_type(destination)),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_validation": source_validation,
        "copy_started": False,
    }
