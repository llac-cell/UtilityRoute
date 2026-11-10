from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.obb import OBBValidator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.data import PairedYOLODataset  # noqa: E402
from utilityroute.data.config import resolve_data_config  # noqa: E402
from utilityroute.data.corruptions import (  # noqa: E402
    COMPOUND_CONDITIONS,
    CONDITIONS,
    SEVERITY_LEVELS,
    SEVERITY_PARAMETERS,
    SINGLE_CONDITIONS,
    apply_condition,
)
from utilityroute.models import DualStreamOBBModel, UtilityRouteOBBModel  # noqa: E402


DATA_YAML = ROOT / "configs/data/dronevehicle_paired.yaml"
MANIFEST = ROOT / "manifests/dronevehicle_local_scene_disjoint_v1.csv"


class CorruptedPairedYOLODataset(PairedYOLODataset):
    """Paired dataset with deterministic pre-letterbox, in-memory corruptions."""

    condition = "clean"
    severity = 0
    corruption_seed = 42

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Build a missing/invalid label cache under a temporary directory, never beside source labels."""
        with tempfile.TemporaryDirectory(prefix="utilityroute_label_cache_") as temp_dir:
            return super().cache_labels(Path(temp_dir) / "labels.cache")

    def load_image(self, i: int, rect_mode: bool = True):
        image, original_shape, resized_shape = super().load_image(i, rect_mode)
        if self.condition == "clean":
            return image, original_shape, resized_shape
        chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        corrupted = apply_condition(
            chw,
            self.condition,
            self.severity,
            sample_id=Path(self.im_files[i]).stem,
            base_seed=self.corruption_seed,
        )
        image = np.ascontiguousarray(corrupted.permute(1, 2, 0).cpu().numpy())
        return image, original_shape, resized_shape


class LimitedDataset(Dataset):
    """Deterministic prefix while preserving the YOLO collate function."""

    def __init__(self, dataset: Dataset, limit: int):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.dataset = dataset
        self.limit = min(limit, len(dataset))
        self.collate_fn = getattr(dataset, "collate_fn")

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        return self.dataset[index]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return slug.strip("-") or "corruption-eval"


def checkpoint_seed(model: torch.nn.Module, weights: Path) -> int | str | None:
    model_args = getattr(model, "args", {})
    value = model_args.get("seed") if isinstance(model_args, dict) else getattr(model_args, "seed", None)
    if value is not None:
        return value
    candidate = weights.resolve().parent.parent.name
    return int(candidate) if candidate.isdigit() else candidate or None


def load_paired_checkpoint(weights: Path) -> torch.nn.Module:
    loaded = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        model = loaded.get("ema") or loaded.get("model")
    else:
        raise TypeError(f"Unsupported checkpoint payload: {type(loaded)}")
    if not isinstance(model, DualStreamOBBModel):
        raise TypeError(f"Expected a paired DualStream/UtilityRoute checkpoint, found {type(model)}")
    return model.float().eval()


def configure_policy(model: torch.nn.Module, policy: str) -> dict[str, Any] | None:
    if not isinstance(model, UtilityRouteOBBModel):
        if policy != "native":
            raise ValueError(f"Policy {policy!r} requires a UtilityRoute checkpoint")
        return None

    model.forced_expert = None
    model.forced_expert_mask = None
    if policy == "router":
        model.force_all_experts = False
    elif policy == "always_on":
        model.force_all_experts = False
        model.forced_expert_mask = torch.ones(4)
    elif policy == "identity":
        model.force_all_experts = False
        model.forced_expert_mask = torch.zeros(4)
    elif policy != "native":
        raise ValueError(policy)
    return {
        "policy": policy,
        "implementation_contract": getattr(model.router, "implementation_contract", "legacy_unverified"),
        "expert_input": getattr(model.router, "expert_input", "legacy_unverified"),
        "top_k": int(model.router.top_k),
        "beta": float(model.router.beta),
        "force_all_experts": bool(model.force_all_experts),
        "forced_expert_mask": (
            None if model.forced_expert_mask is None else [int(value) for value in model.forced_expert_mask.tolist()]
        ),
    }


def dataset_split_image_path(data_yaml: Path, data: dict[str, Any], split: str) -> Path:
    """Resolve an image split from a YOLO data config without relying on module globals."""

    if split not in data:
        raise KeyError(f"Data config {data_yaml} has no {split!r} split")
    root = Path(str(data.get("path", data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    split_path = Path(str(data[split])).expanduser()
    return split_path.resolve() if split_path.is_absolute() else (root / split_path).resolve()


def build_dataset(
    args: argparse.Namespace,
    data_yaml: Path = DATA_YAML,
) -> tuple[CorruptedPairedYOLODataset, dict]:
    data_yaml = Path(getattr(args, "data_config", None) or data_yaml).expanduser().resolve()
    data = resolve_data_config(data_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"Data config must contain a YAML mapping: {data_yaml}")
    image_path = dataset_split_image_path(data_yaml, data, "test")
    hyp = get_cfg(
        overrides={
            "imgsz": args.imgsz,
            "task": "obb",
            "mode": "val",
            "rect": True,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "bgr": 0.0,
        }
    )
    dataset = CorruptedPairedYOLODataset(
        img_path=str(image_path),
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=hyp,
        rect=True,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="zero-shot-corruptions-test: ",
        task="obb",
        classes=None,
        data=data,
        fraction=1.0,
    )
    dataset.corruption_seed = args.corruption_seed
    return dataset, data


def condition_parameters(condition: str, severity: int) -> dict[str, float | int]:
    if condition == "clean":
        return {}
    families = COMPOUND_CONDITIONS.get(condition, (condition,))
    return {family: SEVERITY_PARAMETERS[family][severity - 1] for family in families}


def evaluate_condition(
    model: torch.nn.Module,
    dataset: CorruptedPairedYOLODataset,
    condition: str,
    severity: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float]]:
    dataset.condition = condition
    dataset.severity = severity
    evaluation_dataset: Dataset = dataset if args.limit is None else LimitedDataset(dataset, args.limit)
    dataloader = build_dataloader(
        evaluation_dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
    )
    validator_args = {
        "model": str(args.weights),
        "data": str(getattr(args, "data_config", DATA_YAML)),
        "task": "obb",
        "mode": "val",
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "half": args.half,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "verbose": args.verbose,
    }
    with tempfile.TemporaryDirectory(prefix="utilityroute_corruption_validator_") as temp_dir:
        resolved_yaml = Path(temp_dir) / "data.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolve_data_config(getattr(args, "data_config", DATA_YAML))), encoding="utf-8")
        validator_args["data"] = str(resolved_yaml)
        validator = OBBValidator(
            dataloader=dataloader,
            save_dir=Path(temp_dir),
            args=validator_args,
        )
        raw_stats = validator(model=model)
        precision, recall, ap50, ap50_95 = (float(value) for value in validator.metrics.mean_results())
        speed = {key: float(value) for key, value in validator.speed.items()}
    metrics = {
        "precision": precision,
        "recall": recall,
        "ap50": ap50,
        "ap50_95": ap50_95,
        **{f"validator_{key}": float(value) for key, value in raw_stats.items()},
    }
    return metrics, speed


def evaluation_schedule(args: argparse.Namespace) -> list[tuple[str, int]]:
    schedule: list[tuple[str, int]] = []
    if not args.no_clean:
        schedule.append(("clean", 0))
    schedule.extend((condition, severity) for condition in args.conditions for severity in args.severities)
    return schedule


def write_outputs(run_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_dir / "zero_shot_corruptions.json"
    csv_path = run_dir / "zero_shot_corruptions.csv"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic zero-shot corruption evaluation on the frozen local DroneVehicle test split."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=DATA_YAML)
    parser.add_argument("--policy", choices=("native", "router", "always_on", "identity"), default="native")
    parser.add_argument("--model-label")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=list(SEVERITY_LEVELS))
    parser.add_argument("--corruption-seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/zero_shot_corruptions")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    invalid = sorted(set(args.severities) - set(SEVERITY_LEVELS))
    if invalid:
        parser.error(f"--severities must be drawn from {SEVERITY_LEVELS}; got {invalid}")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions must not contain duplicates")
    if len(set(args.severities)) != len(args.severities):
        parser.error("--severities must not contain duplicates")
    return args


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = load_paired_checkpoint(weights)
    route = configure_policy(model, args.policy)

    dataset, _ = build_dataset(args)
    evaluated_samples = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    checkpoint_hash = sha256(weights)
    model_label = safe_slug(args.model_label or weights.resolve().parents[2].name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{safe_slug(args.tag)}" if args.tag else ""
    run_dir = args.output_root.resolve() / safe_slug(f"{model_label}_{args.policy}{tag}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": 1,
        "protocol": "deterministic_zero_shot_corruptions_v1",
        "is_loco": False,
        "trained_on_test_corruptions": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "checkpoint": {
            "path": str(weights),
            "sha256": checkpoint_hash,
            "type": f"{type(model).__module__}.{type(model).__name__}",
            "seed": checkpoint_seed(model, weights),
            "model_label": model_label,
        },
        "evaluation": {
            "split": "test",
            "data_config": str(args.data_config.expanduser().resolve()),
            "dataset_manifest": str(MANIFEST.resolve()),
            "dataset_manifest_sha256": sha256(MANIFEST) if MANIFEST.is_file() else None,
            "data_config_sha256": sha256(args.data_config.expanduser().resolve()),
            "metric_scale": "unit interval; multiply by 100 for manuscript percentage points",
            "source_samples": len(dataset),
            "evaluated_samples": evaluated_samples,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "half": args.half,
            "corruption_seed": args.corruption_seed,
            "conditions": args.conditions,
            "severities": args.severities,
            "include_clean": not args.no_clean,
        },
        "corruption_protocol": {
            "single_conditions": list(SINGLE_CONDITIONS),
            "compound_conditions": {key: list(value) for key, value in COMPOUND_CONDITIONS.items()},
            "severity_parameters": {key: list(value) for key, value in SEVERITY_PARAMETERS.items()},
            "application_point": "paired image after source resize and before validation letterbox/formatting",
            "rgb_channels": "RGB tensor channels 0:3",
            "thermal_channel": 3,
            "family_modalities": {
                "low_light": "RGB only",
                "overexposure": "RGB only",
                "fog": "RGB only",
                "motion_blur": "RGB only",
                "rgb_noise": "RGB only",
                "thermal_low_contrast": "thermal only",
                "rgbt_translation": "thermal only",
                "occlusion": "shared RGB and thermal coordinates",
            },
            "parameter_semantics": {
                "low_light": "gamma exponent greater than one",
                "overexposure": "gamma exponent less than one",
                "fog": "uniform atmospheric transmission with white airlight A=1",
                "motion_blur": "odd kernel length; deterministic angle sampled in [0, 180)",
                "rgb_noise": "zero-mean Gaussian sigma in 8-bit intensity units",
                "thermal_low_contrast": "contrast multiplier around per-image thermal mean",
                "rgbt_translation": "integer offset applied to each active axis",
                "occlusion": "target area fraction before boundary clipping",
            },
            "translation": (
                "thermal only; deterministic one-of-eight axis/diagonal directions; listed magnitude is applied "
                "to each active axis; bilinear interpolation with BORDER_REPLICATE"
            ),
            "occlusion": "one shared RGB-T per-channel-mean-filled rectangle with deterministic position/aspect",
            "compound_order": {key: list(value) for key, value in COMPOUND_CONDITIONS.items()},
            "sample_rng": "SHA-256(base_seed, sample_id, condition, severity, component_index, family)",
        },
        "route": route,
        "hardware": {
            "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "output_dir": str(run_dir),
    }

    rows: list[dict[str, Any]] = []
    for condition, severity in evaluation_schedule(args):
        metrics, speed = evaluate_condition(model, dataset, condition, severity, args)
        condition_type = "clean" if condition == "clean" else (
            "compound" if condition in COMPOUND_CONDITIONS else "single"
        )
        rows.append(
            {
                "protocol": metadata["protocol"],
                "is_loco": False,
                "model_label": model_label,
                "policy": args.policy,
                "checkpoint_path": str(weights),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_seed": metadata["checkpoint"]["seed"],
                "dataset_manifest_sha256": metadata["evaluation"]["dataset_manifest_sha256"],
                "split": "test",
                "condition_type": condition_type,
                "condition": condition,
                "severity": severity,
                "parameters": json.dumps(condition_parameters(condition, severity), sort_keys=True),
                "corruption_seed": args.corruption_seed,
                "samples": evaluated_samples,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "half": args.half,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ap50": metrics["ap50"],
                "ap50_95": metrics["ap50_95"],
                "preprocess_ms_per_image": speed.get("preprocess"),
                "inference_ms_per_image": speed.get("inference"),
                "postprocess_ms_per_image": speed.get("postprocess"),
            }
        )

    report = {"metadata": metadata, "results": rows}
    json_path, csv_path = write_outputs(run_dir, report, rows)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "protocol": metadata["protocol"],
                "is_loco": False,
                "rows": len(rows),
                "samples_per_row": evaluated_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
