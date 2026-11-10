from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utilityroute.training import UtilityRouteTrainer
from utilityroute.training.protocol import load_method, training_overrides, router_options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("experts", "router"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/data/dronevehicle_paired.yaml")
    parser.add_argument("--method-config", type=Path, default=ROOT / "configs/method_v1.yaml")
    parser.add_argument("--utility-cache", type=Path)
    parser.add_argument("--target-mode", choices=("offline", "online"), default="offline",
                        help="offline=unaugmented cached frames; online=frozen teacher targets for actual augmented frames")
    parser.add_argument("--ablation", choices=("full", "no_utility", "no_cost", "no_identity", "no_balance", "spatial"), default="full")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--lambda-utility", type=float)
    parser.add_argument("--lambda-rank", type=float)
    parser.add_argument("--lambda-balance", type=float)
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--use-spatial-risk", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="0", help="CUDA index, cpu, or mps")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    if args.mode == "router" and args.target_mode == "offline" and args.utility_cache is None:
        parser.error("offline router training requires --utility-cache")
    if args.target_mode == "online" and args.utility_cache is not None:
        parser.error("online targets are measured on augmented frames; do not supply --utility-cache")
    method = load_method(args.method_config)
    overrides = training_overrides(method, args.mode, smoke=args.smoke)
    for field in ("epochs", "batch", "workers"):
        if getattr(args, field) is not None:
            overrides[field] = getattr(args, field)
    if args.mode == "router" and args.target_mode == "offline":
        # Targets were measured on one unaugmented observation, so geometry/mosaic
        # must not silently change the observation to which a target is attached.
        overrides.update(mosaic=0.0, fliplr=0.0, translate=0.0, scale=0.0, close_mosaic=0)
    options = router_options(method)
    presets = {"no_utility": {"lambda_utility": 0.0, "lambda_rank": 0.0},
               "no_cost": {"beta": 0.0}, "no_identity": {"allow_identity": False},
               "no_balance": {"lambda_balance": 0.0}, "spatial": {"use_spatial_risk": True}}
    options.update(presets.get(args.ablation, {}))
    for key in ("beta", "top_k", "lambda_utility", "lambda_rank", "lambda_balance"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.no_identity:
        options["allow_identity"] = False
    if args.use_spatial_risk:
        options["use_spatial_risk"] = True
    run_name = f"utility_{args.mode}_{args.ablation}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    overrides.update(model=str(args.weights.resolve()), data=str(args.data.resolve()), device=args.device,
                     seed=args.seed, project=str(ROOT / "runs" / run_name), name=str(args.seed),
                     freeze=list(range(100)))
    trainer = UtilityRouteTrainer(overrides=overrides, route_mode=args.mode, utility_cache=args.utility_cache,
                                  target_mode=args.target_mode, model_options=options)
    trainer.train()


if __name__ == "__main__":
    main()
