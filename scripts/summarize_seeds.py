from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilityroute.evaluation.seed_summary import PAPER_SEEDS, seed_statistics


def nested_value(report: dict, path: str):
    value = report
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def aggregate(paths: list[Path], metrics: list[str], seed_key: str,
              scale: float = 1.0, allow_partial: bool = False) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(nested_value(report, seed_key)) for report in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate run seeds; each seed must contribute exactly one run")
    if not allow_partial and set(seeds) != set(PAPER_SEEDS):
        raise ValueError(f"Expected the five manuscript seeds {PAPER_SEEDS}; got {seeds}")
    if not metrics:
        raise ValueError("Select at least one metric")
    # Prevent mixing evaluation protocols, datasets or detector/routing variants.
    contexts = []
    for report in reports:
        evaluation = report.get("evaluation", {})
        context = {key: evaluation.get(key) for key in
                   ("split", "modality", "source_samples", "evaluated_samples", "imgsz", "half", "route_policy", "subgroup")}
        context["route"] = report.get("route")
        for key in ("protocol", "policy", "tracker", "temporal", "half", "device", "split", "samples", "repeats", "warmup", "sessions", "calls_per_session"):
            if key in report:
                context[key] = report[key]
        context["checkpoint_type"] = report.get("checkpoint", {}).get("type")
        context["data_source_sha256"] = report.get("data_config_sha256")
        context["data_config_sha256"] = evaluation.get("data_config_sha256")
        context["dataset_manifest_sha256"] = evaluation.get("dataset_manifest_sha256")
        contexts.append(context)
    if any(context != contexts[0] for context in contexts[1:]):
        raise ValueError("Evaluation protocols differ; summarize one configuration at a time")
    return {"schema_version": 1, "unit": "independent seeded run", "seeds": seeds,
            "partial_seed_set": set(seeds) != set(PAPER_SEEDS), "scale": scale,
            "protocol": contexts[0],
            "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "metrics": {metric: {"values": [float(nested_value(r, metric)) * scale for r in reports],
                                 **seed_statistics([float(nested_value(r, metric)) * scale for r in reports])}
                        for metric in metrics}}


def main():
    parser = argparse.ArgumentParser(description="Summarize independent seeded runs with sample SD and Student-t CI")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--metric", action="append", help="Repeat dotted paths, e.g. metrics.ap50_95")
    parser.add_argument("--seed-key", default="checkpoint.seed")
    parser.add_argument("--scale", type=float, default=1.0, help="Use 100 for fractional AP to percentage points")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.reports, args.metric or ["metrics.ap50", "metrics.ap50_95"],
                       args.seed_key, args.scale, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
