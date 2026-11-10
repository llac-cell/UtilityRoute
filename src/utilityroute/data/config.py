"""Resolve local data paths relative to the YAML, never Ultralytics global settings."""
from pathlib import Path
import yaml


def resolve_data_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    root = Path(data.get("path", ".")).expanduser()
    root = root if root.is_absolute() else path.parent / root
    root = root.resolve()
    data["path"] = str(root)
    for split in ("train", "val", "test"):
        if split not in data:
            if split != "test":
                raise ValueError(f"Missing {split} split in {path}")
            continue
        entries = data[split] if isinstance(data[split], list) else [data[split]]
        resolved = []
        for entry in entries:
            item = Path(entry).expanduser()
            item = item if item.is_absolute() else root / item
            if not item.exists():
                raise FileNotFoundError(f"Dataset {split} path does not exist: {item}")
            resolved.append(str(item.resolve()))
        data[split] = resolved if isinstance(data[split], list) else resolved[0]
    if "thermal_root" in data:
        thermal = Path(data["thermal_root"]).expanduser()
        data["thermal_root"] = str((thermal if thermal.is_absolute() else path.parent / thermal).resolve())
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError(f"Missing class names: {path}")
    data["names"] = {int(k): str(v) for k, v in names.items()}
    if sorted(data["names"]) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous and start at zero")
    data["nc"] = len(names)
    data["yaml_file"] = str(path)
    return data
