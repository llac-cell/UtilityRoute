"""Batch-one forward latency on actual paired frames, excluding data loading and NMS."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilityroute.models import DualStreamOBBModel, UtilityRouteOBBModel
from utilityroute.repro import seed_everything, sha256_file
from eval_detection import build_dataset, checkpoint_model, checkpoint_seed


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--policy", choices=("dynamic", "identity", "all"), default="dynamic")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-fuse", action="store_true", help="Retain Conv/BN modules instead of the standard inference fusion")
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--calls", type=int, default=1000, help="Timing calls per session, cycling resident frames")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.calls <= 0 or args.sessions <= 0:
        parser.error("samples/calls/sessions must be positive, warmup nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if args.half and device.type != "cuda":
        parser.error("The manuscript FP16 profiling protocol requires a CUDA device")
    seed_everything(args.seed)
    model, _ = checkpoint_model(args.weights)
    if not isinstance(model, DualStreamOBBModel):
        parser.error("Use a dual-stream or UtilityRoute checkpoint")
    model = model.to(device).eval()
    if not args.no_fuse:
        with torch.inference_mode():
            model.fuse(verbose=False)
    model = model.half() if args.half else model.float()
    if isinstance(model, UtilityRouteOBBModel):
        model.force_all_experts = args.policy == "all"
        model.forced_expert = -1 if args.policy == "identity" else None
        model.forced_expert_mask = None
    elif args.policy != "dynamic":
        parser.error("Dual-stream checkpoint has no expert routing policy")
    dataset, config = build_dataset(paired=True, modality="rgb", split=args.split,
                                    image_size=640, batch=1, data_config=args.data_config, rect=False)
    if not len(dataset):
        raise ValueError("Empty profiling split")

    def frame(index):
        image = dataset[index]["img"].unsqueeze(0).to(device=device,
                    dtype=torch.float16 if args.half else torch.float32) / 255.0
        # Validator uses rectangular batches; the paper profiles square 640 inputs.
        h, w = image.shape[-2:]
        if h > 640 or w > 640:
            raise ValueError(f"Unexpected profile image shape: {tuple(image.shape)}")
        dh, dw = 640 - h, 640 - w
        return F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=114 / 255.0)

    elapsed, active, session_latency = [], [], []
    with torch.inference_mode():
        resident_frames = [frame(index) for index in range(min(args.samples, len(dataset)))]
        synchronize(device)
        for session in range(args.sessions):
            for index in range(args.warmup):
                model(resident_frames[index % len(resident_frames)])
            session_times = []
            for index in range(args.calls):
                image = resident_frames[index % len(resident_frames)]
                synchronize(device)
                start = time.perf_counter()
                model(image)
                synchronize(device)
                elapsed.append((time.perf_counter() - start) * 1000)
                session_times.append(elapsed[-1])
                if isinstance(model, UtilityRouteOBBModel):
                    active.append(float(model.last_route.active_mask.sum().item()))
            session_latency.append(sum(session_times) / len(session_times))
    latency = sum(elapsed) / len(elapsed)
    report = {"schema_version": 1, "checkpoint": {"path": str(args.weights.resolve()),
              "sha256": sha256_file(args.weights), "seed": checkpoint_seed(model)},
              "protocol": "forward_only_square640_batch1; excludes preprocessing and NMS",
              "policy": args.policy, "device": str(device), "half": args.half,
              "conv_bn_fused": not args.no_fuse,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
              "torch": torch.__version__, "data_config": str(config.resolve()),
              "data_config_sha256": sha256_file(config), "split": args.split,
              "samples": min(args.samples, len(dataset)), "calls_per_session": args.calls,
              "sessions": args.sessions, "session_latency_ms": session_latency,
              "warmup": args.warmup, "seed": args.seed,
              "metrics": {"latency_ms": latency, "fps": 1000 / latency,
                          "mean_active_experts": sum(active) / len(active) if active else 0.0},
              "latency_ms_per_forward": elapsed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
