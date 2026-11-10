from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.repro import seed_everything  # noqa: E402
from utilityroute.training.protocol import training_overrides
from utilityroute.training.paired_trainer import LocalOBBTrainer


def load_config(path: Path) -> dict:
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    method_path = (path.parent / experiment["method_config"]).resolve()
    method = yaml.safe_load(method_path.read_text(encoding="utf-8"))
    return {"experiment": experiment, "method": method, "sources": [str(path), str(method_path)]}


def manifest_hash() -> str:
    path = ROOT / "manifests" / "dronevehicle_local_scene_disjoint_v1.csv"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "not-provided"


def serializable_metrics(metrics: dict) -> dict[str, float | str]:
    output: dict[str, float | str] = {}
    for key, value in metrics.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            output[str(key)] = str(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="rgb")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", help="CUDA index, cpu, or mps; defaults to experiment device")
    parser.add_argument("--weights", default="yolov8s-obb.pt")
    parser.add_argument("--data-yaml", type=Path, help="optional staged data YAML; default behavior is unchanged")
    parser.add_argument("--workers", type=int, help="optional worker override; default behavior is unchanged")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")

    resolved = load_config(args.config.resolve())
    experiment = resolved["experiment"]
    method = resolved["method"]
    seed = int(experiment["seed"])
    seed_everything(seed, bool(method["training"]["deterministic"]))

    suffix = f"_{args.modality}" + ("_smoke" if args.smoke else "")
    run_id = experiment["experiment_id"] + suffix
    run_dir = ROOT / "runs" / run_id / str(seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    training = method["training"]
    data_yaml = (
        args.data_yaml.expanduser().resolve()
        if args.data_yaml is not None
        else ROOT / "configs" / "data" / f"dronevehicle_{args.modality}.yaml"
    )
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    workers = 0 if args.smoke else (args.workers if args.workers is not None else int(training["workers"]))
    resolved["runtime"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "modality": args.modality,
        "smoke": args.smoke,
        "weights": args.weights,
        "data_yaml": str(data_yaml),
        "workers": workers,
        "data_manifest_sha256": manifest_hash(),
        "device": args.device or str(experiment.get("device", "0")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    environment = "\n".join(
        sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    ) + "\n"
    (run_dir / "environment.txt").write_text(environment, encoding="utf-8")
    (run_dir / "data_manifest.sha256").write_text(manifest_hash() + "\n", encoding="utf-8")

    model = YOLO(args.weights, task="obb")
    overrides = training_overrides(method, "baseline", smoke=args.smoke)
    overrides.update(data=str(data_yaml), device=args.device or str(experiment.get("device", "0")),
                     workers=workers, seed=seed, project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    results = model.train(trainer=LocalOBBTrainer, **overrides)
    metrics = serializable_metrics(getattr(results, "results_dict", {}))
    (run_dir / "metrics.raw.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
