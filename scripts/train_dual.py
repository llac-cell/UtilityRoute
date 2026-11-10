from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.training import PairedOBBTrainer
from utilityroute.training.protocol import load_method, training_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/data/dronevehicle_paired.yaml")
    parser.add_argument("--method-config", type=Path, default=ROOT / "configs/method_v1.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="0", help="CUDA index, cpu, or mps")
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    method = load_method(args.method_config)
    overrides = training_overrides(method, "dual", smoke=args.smoke)
    for field, argument in (("epochs", args.epochs), ("batch", args.batch), ("workers", args.workers), ("lr0", args.lr)):
        if argument is not None:
            overrides[field] = argument
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"d3_dual_stream{tag}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    overrides.update(model=str(args.weights.resolve()), data=str(args.data.resolve()), device=args.device,
                     seed=args.seed, project=str(ROOT / "runs" / run_name), name=str(args.seed))
    PairedOBBTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()
