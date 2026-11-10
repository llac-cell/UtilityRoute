from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb import OBBValidator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.data import PairedYOLODataset  # noqa: E402
from utilityroute.data.config import resolve_data_config  # noqa: E402
from utilityroute.models import DualStreamOBBModel, UtilityRouteOBBModel  # noqa: E402
from eval_corruptions import configure_policy, dataset_split_image_path  # noqa: E402


SPLITS = ("train", "val", "test")


class LimitedDataset(Dataset):
    """Deterministic prefix of a YOLO dataset while preserving its collate function."""

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



class FrameSubset(Dataset):
    """A deterministic subset selected by explicit paired-frame IDs."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.collate_fn = dataset.collate_fn
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def subgroup_subset(dataset, manifest: Path, group: str, subgroup: str):
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "group", "subgroup"}.issubset(reader.fieldnames or []):
            raise ValueError("Subgroup CSV requires sample_id, group, subgroup columns")
        rows = list(reader)
    selected = {row["sample_id"] for row in rows if row["group"] == group and row["subgroup"] == subgroup}
    if not selected:
        raise ValueError(f"No frames in subgroup {group}/{subgroup}")
    sample_ids = [Path(path).stem for path in dataset.im_files]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Ambiguous basename sample IDs in evaluation split")
    indices = [index for index, sample in enumerate(sample_ids) if sample in selected]
    if not indices:
        raise ValueError("Selected subgroup has no frames in the requested split")
    return FrameSubset(dataset, indices), {
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "group": group, "subgroup": subgroup, "frames": len(indices),
        "sample_ids": [sample_ids[index] for index in indices],
        "filter_scope": "whole frames; object-scale annotation filtering requires a separately defined protocol",
    }

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(prefix: Path) -> tuple[Path, Path]:
    return Path(f"{prefix}.json"), Path(f"{prefix}.csv")


def ensure_outputs_absent(prefix: Path) -> tuple[Path, Path]:
    json_path, csv_path = output_paths(prefix)
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {joined}")
    return json_path, csv_path


def checkpoint_model(weights: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    loaded = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        return loaded.float().eval(), {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(loaded)}")
    model = loaded.get("ema") or loaded.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Checkpoint does not contain a torch model: {weights}")
    return model.float().eval(), loaded


def checkpoint_seed(model: torch.nn.Module) -> int | None:
    model_args = getattr(model, "args", {})
    if not isinstance(model_args, dict):
        try:
            model_args = vars(model_args)
        except TypeError:
            return None
    value = model_args.get("seed")
    return None if value is None else int(value)


def data_paths(paired: bool, modality: str) -> tuple[Path, Path]:
    if paired:
        return ROOT / "configs/data/dronevehicle_paired.yaml", ROOT / "data/dronevehicle_rgb/images"
    return ROOT / f"configs/data/dronevehicle_{modality}.yaml", ROOT / f"data/dronevehicle_{modality}/images"


def build_dataset(
    *,
    paired: bool,
    modality: str,
    split: str,
    image_size: int,
    batch: int,
    data_config: Path | None = None,
    rect: bool = True,
) -> tuple[Dataset, Path]:
    data_yaml = (data_paths(paired, modality)[0] if data_config is None else data_config).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Data configuration does not exist: {data_yaml}")
    data = resolve_data_config(data_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"Data config must contain a mapping: {data_yaml}")
    image_path = dataset_split_image_path(data_yaml, data, split)
    hyp = get_cfg(
        overrides={
            "imgsz": image_size,
            "task": "obb",
            "mode": "val",
            "rect": rect,
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
    dataset_type = PairedYOLODataset if paired else YOLODataset
    dataset = dataset_type(
        img_path=str(image_path),
        imgsz=image_size,
        batch_size=batch,
        augment=False,
        hyp=hyp,
        rect=rect,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5 if rect else 0.0,
        prefix=f"eval-{split}: ",
        task="obb",
        classes=None,
        data=data,
        fraction=1.0,
    )
    return dataset, data_yaml


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def metrics_from_validator(validator: OBBValidator) -> tuple[dict[str, float], list[dict[str, Any]]]:
    precision, recall, ap50, ap50_95 = (float(value) for value in validator.metrics.mean_results())
    overall = {
        "precision": precision,
        "recall": recall,
        "ap50": ap50,
        "ap50_95": ap50_95,
    }

    result_index = {
        int(class_id): result_index
        for result_index, class_id in enumerate(validator.metrics.ap_class_index)
    }
    per_class: list[dict[str, Any]] = []
    for class_id in range(int(validator.nc)):
        class_name = validator.names[class_id] if isinstance(validator.names, dict) else validator.names[class_id]
        metric_index = result_index.get(class_id)
        if metric_index is None:
            values = (None, None, None, None)
        else:
            values = validator.metrics.class_result(metric_index)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "precision": optional_float(values[0]),
                "recall": optional_float(values[1]),
                "ap50": optional_float(values[2]),
                "ap50_95": optional_float(values[3]),
            }
        )
    return overall, per_class


def csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "scope": "overall",
            "class_id": "",
            "class_name": "all",
            **report["metrics"],
        }
    ]
    rows.extend({"scope": "class", **row} for row in report["per_class"])
    return rows


def write_outputs(prefix: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path, csv_path = ensure_outputs_absent(prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        rows = csv_rows(report)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model, _ = checkpoint_model(weights)
    policy = getattr(args, "policy", "native")
    route_policy = configure_policy(model, policy)
    expert_mask = getattr(args, "expert_mask", None)
    if expert_mask is not None:
        if not isinstance(model, UtilityRouteOBBModel):
            raise TypeError("--expert-mask requires a UtilityRoute checkpoint")
        if policy != "native":
            raise ValueError("--expert-mask is mutually exclusive with an explicit --policy")
        model.force_all_experts = False
        model.forced_expert = None
        model.forced_expert_mask = torch.tensor(expert_mask, dtype=torch.float32)
        route_policy = {**(route_policy or {}), "policy": "fixed_mask", "forced_expert_mask": expert_mask}
    paired = isinstance(model, DualStreamOBBModel)
    dataset, data_yaml = build_dataset(
        paired=paired,
        modality=args.modality,
        split=args.split,
        image_size=args.imgsz,
        batch=args.batch,
        data_config=args.data_config,
    )
    source_samples = len(dataset)
    evaluation_dataset: Dataset = dataset
    subgroup = None
    if getattr(args, "subgroup_manifest", None) is not None:
        evaluation_dataset, subgroup = subgroup_subset(dataset, args.subgroup_manifest, args.group, args.subgroup)
    if args.limit is not None:
        evaluation_dataset = LimitedDataset(evaluation_dataset, args.limit)
    dataloader = build_dataloader(
        evaluation_dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
    )

    validator_args = {
        "model": str(weights),
        "data": str(data_yaml),
        "task": "obb",
        "mode": "val",
        "split": args.split,
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
    with tempfile.TemporaryDirectory(prefix="utilityroute_eval_") as temp_dir:
        resolved_yaml = Path(temp_dir) / "data.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolve_data_config(data_yaml)), encoding="utf-8")
        validator_args["data"] = str(resolved_yaml)
        validator = OBBValidator(
            dataloader=dataloader,
            save_dir=Path(temp_dir),
            args=validator_args,
        )
        raw_stats = validator(model=model)
        metrics, per_class = metrics_from_validator(validator)

    manifest = ROOT / "manifests/dronevehicle_local_scene_disjoint_v1.csv"
    route: dict[str, Any] | None = None
    if isinstance(model, UtilityRouteOBBModel):
        route = {
            **(route_policy or {}),
            "implementation_contract": getattr(model.router, "implementation_contract", "legacy_unverified"),
            "expert_input": getattr(model.router, "expert_input", "legacy_unverified"),
            "top_k": int(model.router.top_k),
            "beta": float(model.router.beta),
            "force_all_experts": bool(model.force_all_experts),
            "forced_expert": model.forced_expert,
        }
    hardware = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "checkpoint": {
            "path": str(weights),
            "sha256": sha256(weights),
            "type": f"{type(model).__module__}.{type(model).__name__}",
            "seed": checkpoint_seed(model),
        },
        "evaluation": {
            "split": args.split,
            "modality": "paired_rgb_thermal" if paired else args.modality,
            "data_config": str(data_yaml.resolve()),
            "dataset_manifest": str(manifest.resolve()),
            "dataset_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
            "data_config_sha256": sha256(data_yaml),
            "metric_scale": "unit interval; multiply by 100 for manuscript percentage points",
            "source_samples": source_samples,
            "subgroup": subgroup,
            "evaluated_samples": len(evaluation_dataset),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "half": args.half,
        },
        "route": route,
        "hardware": hardware,
        "metrics": metrics,
        "per_class": per_class,
        "validator_speed_ms_per_image": {key: float(value) for key, value in validator.speed.items()},
        "validator_stats": {str(key): float(value) for key, value in raw_stats.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing RGB, DualStream, or UtilityRoute OBB checkpoint without overwriting evidence."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--policy", choices=("native", "router", "always_on", "identity"), default="native")
    parser.add_argument("--expert-mask", nargs=4, type=int, choices=(0, 1), help="Fixed RGB, thermal, weather, alignment mask; supports single/pair/all expert comparisons")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="rgb")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--data-config",
        type=Path,
        help="optional staged data YAML; default behavior and existing evidence are unchanged",
    )
    parser.add_argument("--subgroup-manifest", type=Path, help="CSV columns sample_id,group,subgroup; selects whole paired frames")
    parser.add_argument("--group")
    parser.add_argument("--subgroup")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if any((args.subgroup_manifest is not None, args.group is not None, args.subgroup is not None)) and not all((args.subgroup_manifest is not None, args.group is not None, args.subgroup is not None)):
        parser.error("--subgroup-manifest, --group, and --subgroup must be used together")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    ensure_outputs_absent(args.output_prefix)
    report = evaluate(args)
    json_path, csv_path = write_outputs(args.output_prefix, report)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "split": report["evaluation"]["split"],
                "samples": report["evaluation"]["evaluated_samples"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
