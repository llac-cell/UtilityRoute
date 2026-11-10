from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.obb.train import OBBTrainer


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_args(args: dict[str, Any], *, omitted_fields: set[str]) -> bytes:
    normalized = dict(args)
    for field in omitted_fields:
        normalized.pop(field, None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _update_semantic_digest(
    digest: Any,
    value: Any,
    path: tuple[str, ...] = (),
    excluded_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> None:
    """Hash checkpoint content semantically while ignoring declared train_args fields."""
    if path in excluded_paths:
        return
    digest.update(type(value).__module__.encode())
    digest.update(type(value).__qualname__.encode())
    if isinstance(value, torch.nn.Module):
        _update_semantic_digest(digest, value.state_dict(), path + ("state_dict",), excluded_paths)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            _update_semantic_digest(digest, value[key], path + (str(key),), excluded_paths)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _update_semantic_digest(digest, item, path + (str(index),), excluded_paths)
    elif isinstance(value, (str, int, float, bool, type(None), Path)):
        digest.update(str(value).encode("utf-8"))
    else:
        digest.update(repr(value).encode("utf-8"))


def semantic_digest_excluding_fields(checkpoint: dict[str, Any], fields: set[str]) -> str:
    digest = hashlib.sha256()
    excluded_paths = frozenset(("train_args", field) for field in fields)
    _update_semantic_digest(digest, checkpoint, excluded_paths=excluded_paths)
    return digest.hexdigest()


def resolve_builtin_resume(checkpoint_path: Path) -> dict[str, object]:
    """Run the exact Ultralytics resume argument resolver without trainer construction or writes."""
    overrides = {
        "task": "obb",
        "mode": "train",
        "model": str(checkpoint_path),
        "resume": str(checkpoint_path),
    }
    trainer = object.__new__(OBBTrainer)
    trainer.args = get_cfg(overrides=overrides)
    OBBTrainer.check_resume(trainer, overrides)
    workers = int(trainer.args.workers)
    return {
        "resume": str(trainer.args.resume),
        "workers": workers,
        "expected_ultralytics_log": f"Using {workers} dataloader workers",
        "project": str(trainer.args.project),
        "name": str(trainer.args.name),
        "data": str(trainer.args.data),
        "epochs": int(trainer.args.epochs),
    }


def prepare_workers_zero_checkpoint(
    source: Path,
    target: Path,
    data_yaml: Path,
    provenance_path: Path | None = None,
    *,
    verify_ultralytics_resume: bool = True,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    data_yaml = data_yaml.expanduser().resolve()
    provenance_path = (
        provenance_path.expanduser().resolve()
        if provenance_path is not None
        else target.with_suffix(target.suffix + ".provenance.json")
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    if source == target:
        raise ValueError("source and target must differ")
    if target.exists() or provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite target/provenance: {target}, {provenance_path}")
    target.parent.mkdir(parents=True, exist_ok=True)

    source_before = source.stat()
    source_hash_before = file_sha256(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_after = source.stat()
    source_hash_after = file_sha256(source)
    if (source_before.st_size, source_before.st_mtime_ns, source_hash_before) != (
        source_after.st_size,
        source_after.st_mtime_ns,
        source_hash_after,
    ):
        raise RuntimeError("source checkpoint changed while it was being read")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("train_args"), dict):
        raise ValueError("checkpoint has no train_args dictionary")

    original_args = dict(checkpoint["train_args"])
    source_workers = original_args.get("workers")
    source_data = original_args.get("data")
    changed_field_names = {"workers", "data"}
    checkpoint["train_args"] = {**original_args, "workers": 0, "data": str(data_yaml)}
    args_without_changed_fields_before = hashlib.sha256(
        _canonical_args(original_args, omitted_fields=changed_field_names)
    ).hexdigest()
    args_without_changed_fields_after = hashlib.sha256(
        _canonical_args(checkpoint["train_args"], omitted_fields=changed_field_names)
    ).hexdigest()
    semantic_before = semantic_digest_excluding_fields(checkpoint, changed_field_names)

    token = f"{os.getpid()}-{time.time_ns()}"
    target_partial = target.with_name(f".{target.stem}.partial-{token}{target.suffix}")
    provenance_partial = provenance_path.with_name(f".{provenance_path.name}.partial-{token}")
    torch.save(checkpoint, target_partial)
    target_hash = file_sha256(target_partial)
    reloaded = torch.load(target_partial, map_location="cpu", weights_only=False)
    semantic_after = semantic_digest_excluding_fields(reloaded, changed_field_names)
    if reloaded["train_args"].get("workers") != 0:
        raise RuntimeError("reloaded checkpoint does not contain workers=0")
    if reloaded["train_args"].get("data") != str(data_yaml):
        raise RuntimeError("reloaded checkpoint does not contain the requested staged data YAML")
    if (
        args_without_changed_fields_before != args_without_changed_fields_after
        or semantic_before != semantic_after
    ):
        raise RuntimeError("checkpoint verification found a semantic change besides train_args.workers/data")

    resume_resolution = resolve_builtin_resume(target_partial) if verify_ultralytics_resume else None
    if resume_resolution is not None:
        if resume_resolution["workers"] != 0:
            raise RuntimeError("Ultralytics built-in resume did not resolve workers=0")
        if resume_resolution["data"] != str(data_yaml):
            raise RuntimeError("Ultralytics built-in resume did not resolve the staged data YAML")
        if resume_resolution["project"] != str(original_args.get("project")):
            raise RuntimeError("Ultralytics resume changed the original project")
        if resume_resolution["name"] != str(original_args.get("name")):
            raise RuntimeError("Ultralytics resume changed the original name")
        # The resolver saw the temporary filename; report the final immutable path instead.
        resume_resolution["resume"] = str(target)

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "copy checkpoint and change only train_args.workers to 0 and train_args.data",
        "source": {
            "path": str(source),
            "size_bytes": source_before.st_size,
            "sha256": source_hash_before,
        },
        "target": {
            "path": str(target),
            "size_bytes": target_partial.stat().st_size,
            "sha256": target_hash,
        },
        "changed_fields": {
            "train_args.workers": {"before": source_workers, "after": 0},
            "train_args.data": {"before": source_data, "after": str(data_yaml)},
        },
        "unchanged_guards": {
            "train_args_excluding_changed_fields_sha256": args_without_changed_fields_before,
            "checkpoint_semantic_sha256_excluding_changed_fields": semantic_after,
            "project": original_args.get("project"),
            "name": original_args.get("name"),
            "original_data": original_args.get("data"),
            "top_level_keys": sorted(str(key) for key in checkpoint),
        },
        "software": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "resume_resolution": resume_resolution,
        "training_started": False,
    }
    with provenance_partial.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    target_partial.replace(target)
    provenance_partial.replace(provenance_path)
    report["provenance_path"] = str(provenance_path)
    return report
