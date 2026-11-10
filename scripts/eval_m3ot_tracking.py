from __future__ import annotations

import argparse
import csv
import importlib.machinery
import json
import math
import shlex
import sys
import time
import types
from datetime import datetime, timezone
from contextlib import nullcontext
import inspect
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import scipy
import torch
import ultralytics
import yaml
from ultralytics.utils import ops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.data.m3ot import (  # noqa: E402
    M3OTFrame,
    audit_m3ot_split,
    group_sequences,
    load_m3ot_manifest,
    load_mot_ground_truth,
    sequence_info,
)
from utilityroute.evaluation import (  # noqa: E402
    TrackingFrame,
    combine_sequence_metrics,
    detection_average_precision,
    tracking_metrics,
)
from utilityroute.models import DualStreamOBBModel, UtilityRouteOBBModel  # noqa: E402
from utilityroute.evaluation.temporal import TemporalUtilityEMA  # noqa: E402
from utilityroute.repro import seed_everything, sha256_file  # noqa: E402


SPLITS = ("train", "val", "test")
FRONTENDS = ("auto", "rgb", "dual", "always-on", "router", "temporal")
DEFAULT_MANIFEST = ROOT / "manifests/m3ot_pairs.csv"
DEFAULT_M3OT_ROOT = ROOT / "data/M3OT"
DEFAULT_TRACKER_CONFIG = Path(ultralytics.__file__).resolve().parent / "cfg/trackers/bytetrack.yaml"
METRIC_REFERENCE = {
    "HOTA": "https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py",
    "Identity": "https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py",
    "CLEAR": "https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py",
}


class ByteTrackDetections:
    """Minimal Results-like adapter accepted by Ultralytics BYTETracker."""

    def __init__(self, xyxy: np.ndarray, scores: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)
        if len(self.xyxy) != len(self.conf):
            raise ValueError("boxes and scores must have equal length")

    @property
    def xywh(self) -> np.ndarray:
        output = self.xyxy.copy()
        output[:, 2:] -= output[:, :2]
        output[:, :2] += output[:, 2:] / 2.0
        return output

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "ByteTrackDetections":
        return ByteTrackDetections(self.xyxy[index].reshape(-1, 4), self.conf[index].reshape(-1))


def load_bytetracker() -> tuple[type, str]:
    """Load Ultralytics BYTETracker and select its built-in SciPy assignment path.

    Ultralytics imports the optional ``lap`` module eagerly. The experiment
    environment intentionally remains unchanged, so a non-callable import shim
    is installed when lap is absent, and every ByteTrack assignment is then
    routed to the matching module's own ``use_lap=False`` SciPy branch.
    """

    shim_created = False
    try:
        __import__("lap")
    except ImportError:
        shim = types.ModuleType("lap")
        shim.__version__ = "scipy-fallback"
        shim.__spec__ = importlib.machinery.ModuleSpec("lap", loader=None)

        def unavailable_lapjv(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("lapjv shim must not be called; SciPy assignment patch is active")

        shim.lapjv = unavailable_lapjv
        sys.modules["lap"] = shim
        shim_created = True

    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.utils import matching
    if shim_created:
        # Keep the import workaround local: OC-SORT should see a real ImportError
        # and select its own SciPy fallback rather than importing our shim.
        sys.modules.pop("lap", None)

    if not getattr(matching.linear_assignment, "_utilityroute_scipy", False):
        original = matching.linear_assignment

        def scipy_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
            del use_lap
            return original(cost_matrix, thresh, use_lap=False)

        scipy_assignment._utilityroute_scipy = True  # type: ignore[attr-defined]
        matching.linear_assignment = scipy_assignment
    return BYTETracker, "scipy.optimize.linear_sum_assignment via ultralytics matching(use_lap=False)"



def load_ocsort(root: Path | None):
    """Import the official OC_SORT checkout supplied explicitly by the caller."""
    if root is None:
        raise ValueError("--tracker ocsort requires --ocsort-root pointing to the official noahcao/OC_SORT checkout")
    root = root.expanduser().resolve()
    source = root / "trackers/ocsort_tracker/ocsort.py"
    if not source.is_file():
        raise FileNotFoundError(f"Expected official OC-SORT module: {source}")
    sys.path.insert(0, str(root))
    try:
        from trackers.ocsort_tracker.ocsort import OCSort
    except ImportError as exc:
        raise RuntimeError("Cannot import OC-SORT; install its official checkout dependencies (including filterpy).") from exc
    if Path(inspect.getfile(OCSort)).resolve() != source:
        raise ImportError("Imported OC-SORT does not belong to --ocsort-root; check conflicting trackers packages")
    return OCSort, source


class OCSortAdapter:
    """Normalize the official Nx5 xyxy,id output to the local tracker adapter."""

    def __init__(self, tracker_type, args, height: int, width: int):
        self.tracker = tracker_type(
            det_thresh=args.ocsort_det_thresh, max_age=args.ocsort_max_age,
            min_hits=args.ocsort_min_hits, iou_threshold=args.ocsort_iou,
            delta_t=args.ocsort_delta_t, inertia=args.ocsort_inertia,
            asso_func="iou", use_byte=False,
        )
        self.image_shape = (height, width)

    def update(self, detections: ByteTrackDetections):
        # Boxes have already been restored to original pixels. Passing equal
        # image/input sizes keeps the official update's rescaling factor at 1.
        values = np.column_stack([detections.xyxy, detections.conf]).astype(np.float32)
        result = self.tracker.update(values.copy(), self.image_shape, self.image_shape)
        if not len(result):
            return np.empty((0, 6), dtype=np.float32)
        # Association metrics ignore confidence; OC-SORT has no score column.
        return np.column_stack([result[:, :5], np.ones(len(result))])

def output_paths(prefix: Path) -> tuple[Path, Path]:
    return Path(f"{prefix}.json"), Path(f"{prefix}.csv")


def ensure_outputs_absent(prefix: Path) -> tuple[Path, Path]:
    json_path, csv_path = output_paths(prefix)
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite tracking output: {', '.join(map(str, existing))}")
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
    args = getattr(model, "args", {})
    if isinstance(args, dict):
        value = args.get("seed")
    else:
        value = getattr(args, "seed", None)
    return None if value is None else int(value)


def configure_frontend(model: torch.nn.Module, requested: str) -> tuple[str, dict[str, Any] | None]:
    if isinstance(model, UtilityRouteOBBModel):
        inferred = "always-on" if bool(model.force_all_experts) else "router"
    elif isinstance(model, DualStreamOBBModel):
        inferred = "dual"
    else:
        inferred = "rgb"
    effective = inferred if requested == "auto" else requested
    if effective == "rgb" and isinstance(model, DualStreamOBBModel):
        raise TypeError("--frontend rgb requires a plain RGB OBB checkpoint")
    if effective == "dual" and (not isinstance(model, DualStreamOBBModel) or isinstance(model, UtilityRouteOBBModel)):
        raise TypeError("--frontend dual requires a DualStream checkpoint without UtilityRoute")
    if effective in {"always-on", "router", "temporal"} and not isinstance(model, UtilityRouteOBBModel):
        raise TypeError(f"--frontend {effective} requires a UtilityRoute checkpoint")
    route = None
    if isinstance(model, UtilityRouteOBBModel):
        original = {
            "force_all_experts": bool(model.force_all_experts),
            "forced_expert": model.forced_expert,
        }
        model.force_all_experts = effective == "always-on"
        model.forced_expert = None
        model.forced_expert_mask = None
        route = {
            "checkpoint_state": original,
            "implementation_contract": getattr(model.router, "implementation_contract", "legacy_unverified"),
            "expert_input": getattr(model.router, "expert_input", "legacy_unverified"),
            "effective_force_all_experts": bool(model.force_all_experts),
            "top_k": int(model.router.top_k),
            "beta": float(model.router.beta),
        }
    return effective, route


def torch_device(value: str) -> torch.device:
    if value.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    index = int(value.removeprefix("cuda:"))
    return torch.device(f"cuda:{index}")


def letterbox_shape(
    height: int, width: int, image_size: int, stride: int = 32
) -> tuple[float, int, int, int, int, int, int]:
    ratio = min(image_size / height, image_size / width)
    resized_height, resized_width = int(round(height * ratio)), int(round(width * ratio))
    output_height = int(math.ceil(resized_height / stride) * stride)
    output_width = int(math.ceil(resized_width / stride) * stride)
    pad_y = (output_height - resized_height) // 2
    pad_x = (output_width - resized_width) // 2
    return ratio, resized_height, resized_width, pad_y, pad_x, output_height, output_width


def _resize_and_pad(
    image: np.ndarray,
    resized_height: int,
    resized_width: int,
    pad_y: int,
    pad_x: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    if image.shape[:2] != (resized_height, resized_width):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    bottom = output_height - resized_height - pad_y
    right = output_width - resized_width - pad_x
    return cv2.copyMakeBorder(image, pad_y, bottom, pad_x, right, cv2.BORDER_CONSTANT, value=114)


def prepare_frame(frame: M3OTFrame, image_size: int, paired: bool) -> tuple[np.ndarray, dict[str, float | int]]:
    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(frame.rgb_path)
    original_height, original_width = rgb.shape[:2]
    ratio, resized_height, resized_width, pad_y, pad_x, output_height, output_width = letterbox_shape(
        original_height, original_width, image_size
    )
    rgb = _resize_and_pad(
        rgb, resized_height, resized_width, pad_y, pad_x, output_height, output_width
    )
    if paired:
        thermal = cv2.imread(str(frame.thermal_path), cv2.IMREAD_GRAYSCALE)
        if thermal is None:
            raise FileNotFoundError(frame.thermal_path)
        thermal = _resize_and_pad(
            thermal, resized_height, resized_width, pad_y, pad_x, output_height, output_width
        )
        image = np.concatenate([rgb[..., ::-1], thermal[..., None]], axis=2)  # explicit R,G,B,T matches paired loader
    else:
        image = rgb[..., ::-1]  # plain Ultralytics RGB checkpoints are evaluated in RGB order
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return chw, {
        "ratio": float(ratio),
        "pad_y": int(pad_y),
        "pad_x": int(pad_x),
        "original_height": int(original_height),
        "original_width": int(original_width),
    }


def restore_boxes(prediction: torch.Tensor, geometry: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    if not len(prediction):
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    xywhr = torch.cat([prediction[:, :4], prediction[:, 6:7]], dim=1).clone()
    ratio = float(geometry["ratio"])
    xywhr[:, 0] = (xywhr[:, 0] - float(geometry["pad_x"])) / ratio
    xywhr[:, 1] = (xywhr[:, 1] - float(geometry["pad_y"])) / ratio
    xywhr[:, 2:4] /= ratio
    corners = ops.xywhr2xyxyxyxy(xywhr)
    boxes = torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)
    boxes[:, [0, 2]].clamp_(0, int(geometry["original_width"]))
    boxes[:, [1, 3]].clamp_(0, int(geometry["original_height"]))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid].float().cpu().numpy(), prediction[valid, 4].float().cpu().numpy()


def chunks(values: list[M3OTFrame], size: int) -> Iterator[list[M3OTFrame]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def detect_batch(
    model: torch.nn.Module,
    frames: list[M3OTFrame],
    *,
    image_size: int,
    paired: bool,
    device: torch.device,
    half: bool,
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    prepared = [prepare_frame(frame, image_size, paired) for frame in frames]
    shapes = {image.shape for image, _ in prepared}
    if len(shapes) != 1:
        raise ValueError(f"A detector batch contains incompatible letterbox shapes: {shapes}")
    tensor = torch.from_numpy(np.stack([image for image, _ in prepared])).to(device, non_blocking=True)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    with torch.inference_mode():
        raw = model(tensor)
        predictions = ops.non_max_suppression(
            raw,
            conf_thres=confidence,
            iou_thres=nms_iou,
            nc=len(model.names),
            multi_label=True,
            agnostic=True,
            max_det=max_detections,
            rotated=True,
        )
    return [restore_boxes(prediction, geometry) for prediction, (_, geometry) in zip(predictions, prepared)]


def tracker_args(config_path: Path) -> argparse.Namespace:
    with config_path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if values.get("tracker_type") != "bytetrack":
        raise ValueError(f"Expected a ByteTrack config: {config_path}")
    return argparse.Namespace(**values)


def gt_arrays(gt: dict[int, list[Any]], frame_number: int) -> tuple[np.ndarray, np.ndarray]:
    objects = [item for item in gt.get(frame_number, []) if item.confidence > 0 and item.class_id == 1]
    ids = np.asarray([item.track_id for item in objects], dtype=np.int64)
    boxes = np.asarray([item.box for item in objects], dtype=np.float64).reshape(-1, 4)
    return ids, boxes


def sequence_report(
    model: torch.nn.Module,
    sequence_frames: list[M3OTFrame],
    args: argparse.Namespace,
    byte_tracker_type: type,
    tracker_config: argparse.Namespace,
    paired: bool,
    device: torch.device,
) -> tuple[dict[str, Any], list[TrackingFrame], list[TrackingFrame]]:
    info = sequence_info(sequence_frames[0].sequence_dir)
    if args.tracker == "bytetrack":
        tracker = byte_tracker_type(tracker_config, frame_rate=int(info["frame_rate"]))
    else:
        first_rgb = cv2.imread(str(sequence_frames[0].rgb_path), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise FileNotFoundError(sequence_frames[0].rgb_path)
        tracker = OCSortAdapter(byte_tracker_type, args, *first_rgb.shape[:2])
    gt = load_mot_ground_truth(sequence_frames[0].sequence_dir)
    track_frames: list[TrackingFrame] = []
    detection_frames: list[TrackingFrame] = []
    inference_seconds = 0.0
    for batch_frames in chunks(sequence_frames, args.batch):
        start = time.perf_counter()
        detections = detect_batch(
            model,
            batch_frames,
            image_size=args.imgsz,
            paired=paired,
            device=device,
            half=args.half,
            confidence=args.det_conf,
            nms_iou=args.nms_iou,
            max_detections=args.max_det,
        )
        inference_seconds += time.perf_counter() - start
        for frame, (boxes, scores) in zip(batch_frames, detections):
            gt_ids, gt_boxes = gt_arrays(gt, frame.frame)
            tracked = tracker.update(ByteTrackDetections(boxes, scores))
            if len(tracked):
                track_boxes = tracked[:, :4]
                track_ids = tracked[:, 4].astype(np.int64)
                track_scores = tracked[:, 5]
            else:
                track_boxes = np.empty((0, 4), dtype=np.float32)
                track_ids = np.empty(0, dtype=np.int64)
                track_scores = np.empty(0, dtype=np.float32)
            track_frames.append(TrackingFrame(gt_ids, gt_boxes, track_ids, track_boxes, track_scores))
            detection_frames.append(
                TrackingFrame(gt_ids, gt_boxes, np.arange(len(boxes), dtype=np.int64), boxes, scores)
            )
    tracking = tracking_metrics(track_frames, threshold=args.match_iou)
    detection = detection_average_precision(detection_frames)
    return (
        {
            "group": sequence_frames[0].group,
            "sequence": sequence_frames[0].sequence,
            "source_frames": int(info["sequence_length"]),
            "evaluated_frames": len(sequence_frames),
            "is_full_sequence": len(sequence_frames) == int(info["sequence_length"]),
            "frame_rate": int(info["frame_rate"]),
            "inference_seconds": float(inference_seconds),
            "metrics": {
                "MOTA": tracking["MOTA"],
                "IDF1": tracking["IDF1"],
                "HOTA": tracking["HOTA"],
                "DetA": tracking["hota_details"]["DetA"],
                "AssA": tracking["hota_details"]["AssA"],
                "IDSW": tracking["clear_details"]["IDSW"],
                "AP50": detection["AP50"],
                "AP50_95": detection["AP50_95"],
            },
            "tracking_details": tracking,
            "detection_details": detection,
        },
        track_frames,
        detection_frames,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    weights = args.weights.resolve()
    manifest = args.manifest.resolve()
    m3ot_root = args.m3ot_root.resolve()
    tracker_config_path = args.tracker_config.resolve()
    required_paths = [weights, manifest] + ([tracker_config_path] if args.tracker == "bytetrack" else [])
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not m3ot_root.is_dir():
        raise FileNotFoundError(m3ot_root)

    all_frames = load_m3ot_manifest(manifest, args.split)
    audit = audit_m3ot_split(all_frames, m3ot_root)
    selected_frames = all_frames[: args.limit] if args.limit is not None else all_frames
    source_frames = len(all_frames)
    is_full_split = len(selected_frames) == source_frames

    model, _ = checkpoint_model(weights)
    seed = checkpoint_seed(model)
    seed_everything(seed if seed is not None else 0, deterministic=True)
    frontend, route = configure_frontend(model, args.frontend)
    if frontend == "temporal" and args.batch != 1:
        raise ValueError("Temporal routing requires batch size one")
    paired = isinstance(model, DualStreamOBBModel)
    device = torch_device(args.device)
    if args.half and device.type != "cuda":
        raise ValueError("--half requires a CUDA device")
    model = model.to(device).eval()
    model = model.half() if args.half else model.float()

    if args.tracker == "bytetrack":
        byte_tracker_type, assignment_backend = load_bytetracker()
        tracker_config = tracker_args(tracker_config_path)
        tracker_metadata = {
            "implementation": "ultralytics.trackers.byte_tracker.BYTETracker",
            "config_path": str(tracker_config_path),
            "config_sha256": sha256_file(tracker_config_path),
            "config": vars(tracker_config),
            "assignment_backend": assignment_backend,
        }
    else:
        byte_tracker_type, source = load_ocsort(args.ocsort_root)
        tracker_config = argparse.Namespace()
        tracker_metadata = {
            "implementation": "trackers.ocsort_tracker.ocsort.OCSort",
            "source": str(source), "source_sha256": sha256_file(source),
            "reference": "https://github.com/noahcao/OC_SORT",
            "config": {key: value for key, value in vars(args).items() if key.startswith("ocsort_") and key != "ocsort_root"},
        }
    selected_by_sequence = group_sequences(selected_frames)
    source_by_sequence = group_sequences(all_frames)
    sequences: list[dict[str, Any]] = []
    sequence_tracking: list[dict[str, object]] = []
    all_detection_frames: list[TrackingFrame] = []
    for key, frames in sorted(selected_by_sequence.items()):
        temporal = TemporalUtilityEMA(model.router.utility, args.temporal_alpha) if frontend == "temporal" else nullcontext()
        with temporal:
            item, _, detection_frames = sequence_report(
                model, frames, args, byte_tracker_type, tracker_config, paired, device,
            )
        item["source_frames"] = len(source_by_sequence[key])
        item["is_full_sequence"] = len(frames) == len(source_by_sequence[key])
        sequences.append(item)
        sequence_tracking.append(item["tracking_details"])
        all_detection_frames.extend(detection_frames)

    aggregate_tracking = combine_sequence_metrics(sequence_tracking)
    aggregate_detection = detection_average_precision(all_detection_frames)
    metrics = {
        "MOTA": aggregate_tracking["MOTA"],
        "IDF1": aggregate_tracking["IDF1"],
        "HOTA": aggregate_tracking["HOTA"],
        "DetA": aggregate_tracking["hota_details"]["DetA"],
        "AssA": aggregate_tracking["hota_details"]["AssA"],
        "IDSW": aggregate_tracking["clear_details"]["IDSW"],
        "AP50": aggregate_detection["AP50"],
        "AP50_95": aggregate_detection["AP50_95"],
    }
    annotation_files = []
    for group, frames in sorted(group_sequences(all_frames).items()):
        del group
        gt_path = frames[0].sequence_dir / "gt/gt.txt"
        annotation_files.append({"path": str(gt_path), "sha256": sha256_file(gt_path)})
    for group in sorted({frame.group for frame in all_frames}):
        path = m3ot_root / "Annotations" / group / "rgb" / f"{args.split}_cocoformat.json"
        annotation_files.append({"path": str(path), "sha256": sha256_file(path)})

    eligible = bool(is_full_split and audit["status"] == "pass")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "run_kind": "full_split" if is_full_split else "smoke_subset",
        "eligible_for_manuscript_table": False,
        "complete_single_seed_protocol": eligible,
        "checkpoint": {
            "path": str(weights),
            "sha256": sha256_file(weights),
            "type": f"{type(model).__module__}.{type(model).__name__}",
            "seed": seed,
        },
        "evaluation": {
            "dataset": "M3OT",
            "split": args.split,
            "frontend_requested": args.frontend,
            "frontend": frontend,
            "input_modality": "paired_rgb_thermal" if paired else "rgb",
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "m3ot_root": str(m3ot_root),
            "source_frames": source_frames,
            "evaluated_frames": len(selected_frames),
            "limit": args.limit,
            "is_full_split": is_full_split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "half": args.half,
            "det_conf": args.det_conf,
            "nms_iou": args.nms_iou,
            "agnostic_nms": True,
            "max_det": args.max_det,
            "class_mapping": "all five DroneVehicle detector classes -> M3OT vehicle",
            "box_mapping": "oriented predictions -> enclosing axis-aligned xyxy",
            "ground_truth_authority": "native RGB MOT gt/gt.txt floating-point boxes",
        },
        "route": route,
        "audit": audit,
        "annotation_files": annotation_files,
        "tracker": {**tracker_metadata, "match_iou_for_CLEAR_and_ID": args.match_iou},
        "temporal_extension": {
            "enabled": frontend == "temporal",
            "algorithm": "EMA of predicted four-expert utilities before cost-adjusted Top-2 routing",
            "alpha_current_frame": args.temporal_alpha,
            "reset": "each sequence boundary",
            "historical_reproduction": False,
            "specification_note": "The manuscript gives no temporal state equation; this is an explicit configurable implementation choice, not a recovered historical implementation.",
            "core_expert_count": 4,
        },
        "metric_implementation": {
            "local_path": str((ROOT / "src/utilityroute/evaluation/tracking_metrics.py").resolve()),
            "local_sha256": sha256_file(ROOT / "src/utilityroute/evaluation/tracking_metrics.py"),
            "definitions": "Local CLEAR, Identity, and HOTA formulas checked against TrackEval; HOTA averaged over 0.05:0.05:0.95; dataset-specific ignore/distractor preprocessing is not included",
            "references": METRIC_REFERENCE,
            "detection_ap": "one-class COCO-style 101-point interpolation at IoU 0.50:0.05:0.95",
            "metric_scale": "unit interval; MOTA may be negative",
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "metrics": metrics,
        "tracking_details": aggregate_tracking,
        "detection_details": aggregate_detection,
        "sequences": sequences,
        "elapsed_seconds": float(time.perf_counter() - started),
        "limitations": [
            "Cross-dataset evaluation uses DroneVehicle-trained checkpoints without M3OT fine-tuning.",
            "All source detector subclasses are mapped to M3OT's single vehicle class.",
            "M3OT axis-aligned metrics use enclosing boxes for oriented detector outputs.",
            "M3OT has modality-specific annotations; RGB MOT boxes define the reported coordinate system and IR GT is not merged.",
            "Subset/limit runs are smoke diagnostics and are not eligible for manuscript tables.",
        ],
    }


def csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = report["evaluation"]

    def row(scope: str, group: str, sequence: str, source: int, evaluated: int, full: bool, item: dict[str, Any]):
        details = item["tracking_details"]
        detection = item["detection_details"]
        return {
            "scope": scope,
            "group": group,
            "sequence": sequence,
            "split": evaluation["split"],
            "frontend": evaluation["frontend"],
            "source_frames": source,
            "evaluated_frames": evaluated,
            "is_full_split_or_sequence": full,
            "eligible_for_manuscript_table": report["eligible_for_manuscript_table"] if scope == "overall" else False,
            "MOTA": details["MOTA"],
            "IDF1": details["IDF1"],
            "HOTA": details["HOTA"],
            "DetA": details["hota_details"]["DetA"],
            "AssA": details["hota_details"]["AssA"],
            "AP50": detection["AP50"],
            "AP50_95": detection["AP50_95"],
            "CLR_TP": details["clear_details"]["CLR_TP"],
            "CLR_FN": details["clear_details"]["CLR_FN"],
            "CLR_FP": details["clear_details"]["CLR_FP"],
            "IDSW": details["clear_details"]["IDSW"],
            "IDTP": details["identity_details"]["IDTP"],
            "IDFN": details["identity_details"]["IDFN"],
            "IDFP": details["identity_details"]["IDFP"],
            "gt_detections": detection["ground_truth_detections"],
            "predicted_detections": detection["predicted_detections"],
        }

    rows = [
        row(
            "overall",
            "all",
            "all",
            evaluation["source_frames"],
            evaluation["evaluated_frames"],
            evaluation["is_full_split"],
            report,
        )
    ]
    rows.extend(
        row(
            "sequence",
            item["group"],
            item["sequence"],
            item["source_frames"],
            item["evaluated_frames"],
            item["is_full_sequence"],
            item,
        )
        for item in report["sequences"]
    )
    return rows


def write_outputs(prefix: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path, csv_path = ensure_outputs_absent(prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    rows = csv_rows(report)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing DroneVehicle checkpoints on M3OT detection and ByteTrack metrics."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frontend", choices=FRONTENDS, default="auto")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--m3ot-root", type=Path, default=DEFAULT_M3OT_ROOT)
    parser.add_argument("--tracker", choices=("bytetrack", "ocsort"), default="bytetrack")
    parser.add_argument("--ocsort-root", type=Path)
    parser.add_argument("--ocsort-det-thresh", type=float, default=0.5)
    parser.add_argument("--ocsort-max-age", type=int, default=30)
    parser.add_argument("--ocsort-min-hits", type=int, default=3)
    parser.add_argument("--ocsort-iou", type=float, default=0.3)
    parser.add_argument("--ocsort-delta-t", type=int, default=3)
    parser.add_argument("--ocsort-inertia", type=float, default=0.2)
    parser.add_argument("--temporal-alpha", type=float, default=0.5)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--det-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.frontend == "temporal" and args.batch != 1:
        parser.error("--frontend temporal requires --batch 1 for causal sequence order")
    if not 0 < args.temporal_alpha <= 1:
        parser.error("--temporal-alpha must be in (0, 1]")
    if args.tracker == "ocsort" and args.ocsort_root is None:
        parser.error("--tracker ocsort requires --ocsort-root")
    for name in ("ocsort_det_thresh", "ocsort_iou"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in ("imgsz", "batch", "max_det", "ocsort_max_age", "ocsort_min_hits", "ocsort_delta_t"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for name in ("det_conf", "nms_iou", "match_iou"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.limit is not None and "smoke" not in args.output_prefix.name.lower():
        parser.error("--limit outputs must include 'smoke' in --output-prefix")
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
                "run_kind": report["run_kind"],
                "eligible_for_manuscript_table": report["eligible_for_manuscript_table"],
                "split": report["evaluation"]["split"],
                "frontend": report["evaluation"]["frontend"],
                "frames": report["evaluation"]["evaluated_frames"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
