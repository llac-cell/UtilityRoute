"""Manuscript training settings, shared by all command-line entry points."""
from __future__ import annotations

from pathlib import Path
import math
import yaml


def load_method(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("training"), dict):
        raise ValueError("Method config must contain a training mapping")
    return value


def training_overrides(method: dict, stage: str, *, smoke: bool = False) -> dict:
    if stage not in {"dual", "experts", "router", "baseline"}:
        raise ValueError(f"Unknown training stage: {stage}")
    train = method["training"]
    special = stage in {"experts", "router"}
    result = {
        "epochs": int(train.get("stage_epochs", {}).get(stage, 30 if special else train.get("epochs", 100))),
        "batch": int(train.get("batch", 12)), "workers": int(train.get("workers", 8)),
        "imgsz": 640, "optimizer": str(train.get("optimizer", "AdamW")),
        "lr0": float(train.get("router_learning_rate", 0.0005) if stage == "router" else train.get("learning_rate", 0.001)),
        "lrf": float(train.get("lrf", 0.01)), "cos_lr": False,
        "momentum": float(train.get("momentum", 0.937)),
        "weight_decay": float(train.get("weight_decay", 0.0005)),
        "warmup_epochs": float(train.get("utility_warmup_epochs", 2) if special else train.get("warmup_epochs", 3)),
        "patience": int(train.get("utility_patience", 10) if special else train.get("patience", 20)),
        "amp": bool(train.get("amp", True)), "deterministic": bool(train.get("deterministic", True)),
        "fliplr": 0.5, "flipud": 0.0, "translate": 0.1, "scale": 0.5, "mosaic": 1.0,
        "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "erasing": 0.0,
        "close_mosaic": 10, "cache": False, "plots": False, "save": True, "val": True,
        "exist_ok": False, "fraction": 1.0,
    }
    if float(train.get("max_grad_norm", 10.0)) != 10.0:
        raise ValueError("Pinned Ultralytics trainer clips global gradient norm at 10.0")
    if smoke:
        result.update(epochs=1, batch=2, nbs=2, workers=0, warmup_epochs=0, patience=1, amp=False,
                      fraction=0.02, close_mosaic=0)
    return result


def router_options(method: dict) -> dict:
    router, model = method.get("router", {}), method.get("model", {})
    fixed = {"normalized_cost": [0.55, 0.55, 0.75, 1.0],
             "utility_epsilon": 1e-6, "ranking_ignore_threshold": 0.01,
             "ranking_margin": 0.05}
    for key, expected in fixed.items():
        if key not in router:
            continue
        actual = router[key]
        if isinstance(expected, list):
            matches = isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
                math.isclose(float(a), b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(actual, expected))
        else:
            matches = math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"{key} is fixed by the implemented manuscript contract at {expected}; update the implementation before changing it")
    return {
        "beta": float(router.get("beta", 0.015)), "top_k": int(model.get("top_k", 2)),
        "lambda_utility": float(router.get("lambda_utility", 1.0)),
        "lambda_rank": float(router.get("lambda_rank", 0.1)),
        "lambda_balance": float(router.get("lambda_balance", 0.01)),
        "allow_identity": bool(router.get("allow_identity", True)),
        "use_spatial_risk": bool(router.get("use_spatial_risk", False)),
        "spatial_grid_size": int(router.get("spatial_grid_size", 4)),
    }
