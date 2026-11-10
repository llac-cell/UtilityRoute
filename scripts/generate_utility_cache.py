from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch
import yaml
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPLITS = ("train", "val", "test")
DEFAULT_SPLITS = ("train", "val")

from utilityroute.data import PairedYOLODataset  # noqa: E402
from utilityroute.models import UtilityRouteOBBModel  # noqa: E402
from utilityroute.data.config import resolve_data_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_splits(splits: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not splits:
        raise ValueError("At least one split is required")
    invalid = [split for split in splits if split not in SPLITS]
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}")
    return tuple(dict.fromkeys(splits))


def ensure_output_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing utility cache: {path}")


def dataset(split: str, data_config: Path) -> PairedYOLODataset:
    cfg = get_cfg(
        overrides={
            "imgsz": 640,
            "task": "obb",
            "rect": False,
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
    data_config = data_config.expanduser().resolve()
    data = resolve_data_config(data_config)
    image_path = Path(data[split])
    return PairedYOLODataset(
        img_path=str(image_path.resolve()),
        imgsz=640,
        batch_size=1,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=f"utility-{split}: ",
        task="obb",
        classes=None,
        data=data,
        fraction=1.0,
    )


def to_device_batch(item: dict, device: torch.device | str = "cpu") -> dict:
    batch = PairedYOLODataset.collate_fn([item])
    for key in ("img", "batch_idx", "cls", "bboxes"):
        batch[key] = batch[key].to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255.0
    return batch


def to_cuda_batch(item: dict) -> dict:
    """Backward-compatible CUDA-only alias; new callers should pass an explicit device."""
    return to_device_batch(item, "cuda")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/utility_cache_v1.csv")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=ROOT / "configs/data/dronevehicle_paired.yaml",
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--device", default="0", help="CUDA index, cpu, or mps")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.splits = normalize_splits(args.splits)
    return args


def main() -> None:
    args = parse_args()
    ensure_output_absent(args.output)

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, UtilityRouteOBBModel):
        raise TypeError(f"Expected UtilityRouteOBBModel checkpoint, found {type(model)}")
    from ultralytics.utils.torch_utils import select_device
    device = select_device(args.device)
    model = model.float().to(device).eval()
    model.requires_grad_(False)
    # Ultralytics serializes `model.args` as a plain dict in stripped checkpoints,
    # while v8DetectionLoss accesses the hyperparameters as attributes.
    if isinstance(model.args, dict):
        model.args = get_cfg(overrides=model.args)
    model.force_all_experts = False
    model.forced_expert_mask = None
    model.criterion = model.init_criterion()
    checkpoint_hash = sha256(args.weights)

    rows: list[dict] = []
    with torch.inference_mode():
        for split in args.splits:
            data = dataset(split, args.data_config)
            for index in range(len(data)):
                if args.limit is not None and len(rows) >= args.limit:
                    break
                item = data[index]
                batch = to_device_batch(item, device)
                losses: list[float] = []
                for expert in (-1, 0, 1, 2, 3):
                    model.forced_expert = expert
                    predictions = model.predict(batch["img"])
                    loss_components, _ = model.criterion(predictions, batch)
                    losses.append(float(loss_components.detach().sum().cpu()))
                identity = losses[0]
                utilities = [(identity - value) / (identity + 1e-6) for value in losses[1:]]
                row = {
                    "sample_id": Path(item["im_file"]).stem,
                    "image_path": str(Path(item["im_file"]).resolve()),
                    "preprocessing": "rgbt_square640_no_augmentation_v2",
                    "split": split,
                    "identity_loss": identity,
                    **{f"expert_{i}_loss": losses[i + 1] for i in range(4)},
                    **{f"utility_{i}": utilities[i] for i in range(4)},
                    "checkpoint_sha256": checkpoint_hash,
                }
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"generated {len(rows)} utility rows")
            if args.limit is not None and len(rows) >= args.limit:
                break

    model.forced_expert = None
    if not rows:
        raise RuntimeError("Utility cache generation produced no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "rows": len(rows),
                "splits": list(args.splits),
                "checkpoint": str(args.weights),
                "checkpoint_sha256": checkpoint_hash,
                "data_config": str(args.data_config.expanduser().resolve()),
                "data_config_sha256": sha256(args.data_config.expanduser().resolve()),
                "command": shlex.join([sys.executable, *sys.argv]),
                "device": str(device),
                "preprocessing": "rgbt_square640_no_augmentation_v2",
                "target_formula": "(identity_loss-expert_loss)/(identity_loss+1e-6)",
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
