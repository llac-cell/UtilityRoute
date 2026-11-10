from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import select_device


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_corruptions import (  # noqa: E402
    DATA_YAML,
    MANIFEST,
    CorruptedPairedYOLODataset,
    build_dataset,
    checkpoint_seed,
    condition_parameters,
    dataset_split_image_path,
    load_paired_checkpoint,
)
from utilityroute.data.corruptions import (  # noqa: E402
    COMPOUND_CONDITIONS,
    CONDITIONS,
    SEVERITY_LEVELS,
    SEVERITY_PARAMETERS,
    SINGLE_CONDITIONS,
)
from utilityroute.evaluation.condition_routing import (  # noqa: E402
    NativeRouteObservation,
    RoutingObservation,
    action_space,
    add_clean_comparisons,
    condition_sensitivity_summary,
    summarize_condition,
    summarize_native_routes,
)
from utilityroute.models import UtilityRouteOBBModel  # noqa: E402
from utilityroute.repro import seed_everything, sha256_file  # noqa: E402


PROTOCOL = "condition_wise_routing_corruptions_v1"
DEFAULT_WEIGHTS = ROOT / "runs/utility_router_seed42/42/weights/best.pt"
EVIDENCE_MODES = ("full_matrix", "two_stage")


def output_paths(prefix: Path) -> tuple[Path, Path]:
    return Path(f"{prefix}.json"), Path(f"{prefix}.csv")


def ensure_outputs_absent(prefix: Path) -> tuple[Path, Path]:
    paths = output_paths(prefix)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite condition-routing output: {', '.join(map(str, existing))}")
    return paths


def stage_provenance_metadata(data_config: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    """Hash staged-data provenance files when the selected YAML belongs to a stage."""

    data_config = data_config.expanduser().resolve()
    data_root = Path(str(data.get("path", data_config.parent))).expanduser()
    if not data_root.is_absolute():
        data_root = data_config.parent / data_root
    candidates = [data_config.parent.parent, data_root.resolve().parent]
    workspace_manifest_hash = sha256_file(MANIFEST) if MANIFEST.is_file() else None
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        inventory_path = candidate / "provenance/inventory.csv"
        report_path = candidate / "provenance/stage_report.json"
        if not inventory_path.exists() and not report_path.exists():
            continue
        if not inventory_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                f"Incomplete staged-data provenance under {candidate}: "
                "both provenance/inventory.csv and provenance/stage_report.json are required"
            )
        stage_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(stage_report, dict):
            raise ValueError(f"Stage report must contain a JSON object: {report_path}")
        reported_destination = stage_report.get("destination")
        if reported_destination is not None and Path(str(reported_destination)).expanduser().resolve() != candidate:
            raise ValueError(
                f"Stage report destination {reported_destination!r} does not match detected stage root {candidate}"
            )
        reported_manifest_hash = stage_report.get("manifest_sha256")
        return {
            "detected": True,
            "stage_root": str(candidate),
            "inventory": {
                "path": str(inventory_path),
                "file_sha256": sha256_file(inventory_path),
                "reported_content_sha256": stage_report.get("inventory_sha256"),
                "reported_rows": stage_report.get("inventory_rows"),
            },
            "report": {
                "path": str(report_path),
                "file_sha256": sha256_file(report_path),
                "schema_version": stage_report.get("schema_version"),
            },
            "reported_manifest_sha256": reported_manifest_hash,
            "workspace_manifest_sha256": workspace_manifest_hash,
            "manifest_matches_workspace": reported_manifest_hash == workspace_manifest_hash,
        }
    return {
        "detected": False,
        "stage_root": None,
        "inventory": None,
        "report": None,
        "reported_manifest_sha256": None,
        "workspace_manifest_sha256": workspace_manifest_hash,
        "manifest_matches_workspace": None,
    }


def normalize_unique(values: list[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def condition_schedule(
    conditions: list[str] | tuple[str, ...],
    severities: list[int] | tuple[int, ...],
    *,
    include_clean: bool,
) -> list[tuple[str, int]]:
    normalized_conditions = normalize_unique(list(conditions))
    normalized_severities = normalize_unique([int(value) for value in severities])
    invalid_conditions = [value for value in normalized_conditions if value not in CONDITIONS]
    invalid_severities = [value for value in normalized_severities if value not in SEVERITY_LEVELS]
    if invalid_conditions:
        raise ValueError(f"Unknown corruption conditions: {invalid_conditions}")
    if invalid_severities:
        raise ValueError(f"Unknown corruption severities: {invalid_severities}")
    schedule = [("clean", 0)] if include_clean else []
    schedule.extend((condition, severity) for condition in normalized_conditions for severity in normalized_severities)
    if not schedule:
        raise ValueError("The condition schedule is empty")
    return schedule


def full_protocol_arguments(args: argparse.Namespace) -> bool:
    return bool(
        args.limit is None
        and tuple(args.conditions) == tuple(CONDITIONS)
        and tuple(args.severities) == tuple(SEVERITY_LEVELS)
        and args.corruption_seed == 42
        and args.imgsz == 640
        and args.batch == 8
    )


def checkpoint_seed_int(model: torch.nn.Module, weights: Path) -> int | str | None:
    value = checkpoint_seed(model, weights)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def device_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = CorruptedPairedYOLODataset.collate_fn([item])
    for key in ("img", "batch_idx", "cls", "bboxes"):
        batch[key] = batch[key].to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255.0
    return batch


def task_loss(model: UtilityRouteOBBModel, batch: dict[str, Any]) -> float:
    predictions = model.predict(batch["img"])
    loss, _ = model.criterion(predictions, batch)
    return float(loss.detach().sum().cpu())


def evaluate_observation(
    model: UtilityRouteOBBModel,
    batch: dict[str, Any],
    actions: Mapping[str, tuple[int, ...]],
) -> RoutingObservation:
    """Run the native route and every forced action on one corrupted sample."""

    model.forced_expert = None
    model.forced_expert_mask = None
    model.force_all_experts = False
    predictions = model.predict(batch["img"])
    if model.last_route is None:
        raise RuntimeError("UtilityRoute model did not expose last_route")
    route = model.last_route
    predicted_utility = route.predicted_utility[0].detach().float().cpu().numpy()
    net_score = route.net_score[0].detach().float().cpu().numpy()
    active_mask = route.active_mask[0].detach().bool().cpu().numpy()
    routed_loss_tensor, _ = model.criterion(predictions, batch)
    routed_loss = float(routed_loss_tensor.detach().sum().cpu())

    losses: dict[str, float] = {}
    try:
        for label, mask in actions.items():
            model.forced_expert_mask = torch.tensor(mask, device=batch["img"].device, dtype=batch["img"].dtype)
            losses[label] = task_loss(model, batch)
    finally:
        model.forced_expert_mask = None
        model.forced_expert = None
        model.force_all_experts = False
    return RoutingObservation(
        predicted_utility=predicted_utility,
        net_score=net_score,
        active_mask=active_mask,
        identity_loss=losses["identity"],
        routed_loss=routed_loss,
        action_losses=losses,
    )


def evaluate_native_observation(
    model: UtilityRouteOBBModel,
    batch: dict[str, Any],
) -> NativeRouteObservation:
    """Run only the native route for activation/predicted-utility evidence."""

    model.forced_expert = None
    model.forced_expert_mask = None
    model.force_all_experts = False
    model.predict(batch["img"])
    if model.last_route is None:
        raise RuntimeError("UtilityRoute model did not expose last_route")
    route = model.last_route
    return NativeRouteObservation(
        predicted_utility=route.predicted_utility[0].detach().float().cpu().numpy(),
        net_score=route.net_score[0].detach().float().cpu().numpy(),
        active_mask=route.active_mask[0].detach().bool().cpu().numpy(),
    )


def utility_probe(condition: str, severity: int, evidence_mode: str) -> bool:
    if evidence_mode == "full_matrix":
        return True
    if evidence_mode != "two_stage":
        raise ValueError(f"Unknown evidence mode: {evidence_mode}")
    return condition == "clean" or severity == max(SEVERITY_LEVELS)


def condition_type(condition: str) -> str:
    if condition == "clean":
        return "clean"
    return "compound" if condition in COMPOUND_CONDITIONS else "single"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data_config = args.data_config.expanduser().resolve()
    if not data_config.is_file():
        raise FileNotFoundError(data_config)
    model = load_paired_checkpoint(weights)
    if not isinstance(model, UtilityRouteOBBModel):
        raise TypeError(f"Expected UtilityRouteOBBModel checkpoint, found {type(model)}")
    seed = checkpoint_seed_int(model, weights)
    seed_everything(int(seed) if isinstance(seed, int) else 0, deterministic=True)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model.float().to(device).eval()
    if isinstance(model.args, dict):
        model.args = get_cfg(overrides=model.args)
    model.criterion = model.init_criterion()
    model.forced_expert = None
    model.forced_expert_mask = None
    model.force_all_experts = False

    dataset, data = build_dataset(args, data_yaml=data_config)
    stage_provenance = stage_provenance_metadata(data_config, data)
    test_image_path = dataset_split_image_path(data_config, data, "test")
    source_samples = len(dataset)
    evaluated_samples = min(source_samples, args.limit) if args.limit is not None else source_samples
    schedule = condition_schedule(args.conditions, args.severities, include_clean=True)
    names = tuple(model.router.expert_names)
    actions = action_space(names, int(model.router.top_k))
    rows: list[dict[str, Any]] = []
    utility_condition_keys: list[str] = []

    with torch.inference_mode():
        for condition, severity in schedule:
            condition_started = time.perf_counter()
            dataset.condition = condition
            dataset.severity = severity
            has_utility = utility_probe(condition, severity, args.evidence_mode)
            if has_utility:
                observations = [
                    evaluate_observation(model, device_batch(dataset[index], device), actions)
                    for index in range(evaluated_samples)
                ]
                summary = summarize_condition(observations, names, actions)
            else:
                native_observations = [
                    evaluate_native_observation(model, device_batch(dataset[index], device))
                    for index in range(evaluated_samples)
                ]
                summary = summarize_native_routes(native_observations, names)
            condition_key = "clean" if condition == "clean" else f"{condition}:s{severity}"
            if has_utility:
                utility_condition_keys.append(condition_key)
            summary.update(
                {
                    "condition_key": condition_key,
                    "condition_type": condition_type(condition),
                    "condition": condition,
                    "severity": severity,
                    "parameters": condition_parameters(condition, severity),
                    "corruption_seed": args.corruption_seed,
                    "utility_evidence_available": has_utility,
                    "elapsed_seconds": float(time.perf_counter() - condition_started),
                }
            )
            rows.append(summary)

    add_clean_comparisons(rows, names)
    sensitivity = condition_sensitivity_summary(rows, names)
    full = bool(
        full_protocol_arguments(args)
        and evaluated_samples == source_samples
        and len(rows) == 56
        and "smoke" not in args.output_prefix.name.lower()
    )
    corruption_source = ROOT / "src/utilityroute/data/corruptions.py"
    analysis_source = ROOT / "src/utilityroute/evaluation/condition_routing.py"
    native_forwards = len(rows) * evaluated_samples
    forced_forwards = len(utility_condition_keys) * evaluated_samples * len(actions)
    total_forwards = native_forwards + forced_forwards
    formal_run_kind = "full_matrix_protocol" if args.evidence_mode == "full_matrix" else "two_stage_protocol"
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "run_kind": formal_run_kind if full else "smoke_subset",
        "eligible_for_manuscript_table": False,
        "complete_single_seed_protocol": full,
        "checkpoint": {
            "path": str(weights),
            "sha256": sha256_file(weights),
            "type": f"{type(model).__module__}.{type(model).__name__}",
            "seed": seed,
        },
        "evaluation": {
            "split": "test",
            "data_config": str(data_config),
            "data_config_sha256": sha256_file(data_config),
            "resolved_test_image_path": str(test_image_path),
            "stage_provenance": stage_provenance,
            "dataset_manifest": str(MANIFEST.resolve()),
            "dataset_manifest_sha256": sha256_file(MANIFEST) if MANIFEST.is_file() else None,
            "source_samples": source_samples,
            "evaluated_samples_per_condition": evaluated_samples,
            "limit": args.limit,
            "conditions": list(args.conditions),
            "severities": list(args.severities),
            "include_clean": True,
            "condition_rows": len(rows),
            "expected_full_condition_rows": 56,
            "evidence_mode": args.evidence_mode,
            "utility_condition_rows": len(utility_condition_keys),
            "utility_condition_keys": utility_condition_keys,
            "imgsz": args.imgsz,
            "dataset_batch_shape": args.batch,
            "inference_batch": 1,
            "precision": "fp32",
            "corruption_seed": args.corruption_seed,
        },
        "corruption_protocol": {
            "single_conditions": list(SINGLE_CONDITIONS),
            "compound_conditions": {key: list(value) for key, value in COMPOUND_CONDITIONS.items()},
            "severity_parameters": {key: list(value) for key, value in SEVERITY_PARAMETERS.items()},
            "source_path": str(corruption_source.resolve()),
            "source_sha256": sha256_file(corruption_source),
            "application_point": "same CorruptedPairedYOLODataset pre-letterbox path as eval_corruptions.py",
        },
        "router": {
            "implementation_contract": getattr(model.router, "implementation_contract", "legacy_unverified"),
            "expert_input": getattr(model.router, "expert_input", "legacy_unverified"),
            "expert_names": list(names),
            "top_k": int(model.router.top_k),
            "beta": float(model.router.beta),
            "allow_identity": bool(getattr(model.router, "allow_identity", True)),
            "use_spatial_risk": bool(getattr(model.router, "use_spatial_risk", False)),
            "action_space": {label: list(mask) for label, mask in actions.items()},
            "action_count": len(actions),
        },
        "utility_evidence": {
            "definition": "(identity task loss - action task loss) / (identity task loss + 1e-6)",
            "loss_scope": "per-sample OBB task loss with batch size one",
            "true_utility": "recomputed on the same corrupted sample for identity and every mask up to router Top-k",
            "best_action": "minimum task loss over the declared action space; exact ties prefer sparser masks",
            "no_harm": "native routed task loss <= identity task loss",
            "top1": "native highest-score expert/identity versus the true best identity-or-single action",
            "top2": "native active mask covers the true best identity-or-single action",
            "regret": "best identity-or-single relative utility minus native routed relative utility",
            "complete_action_regret": "best-action relative utility minus native routed relative utility",
            "analysis_source": str(analysis_source.resolve()),
            "analysis_source_sha256": sha256_file(analysis_source),
            "evidence_mode": args.evidence_mode,
            "two_stage_definition": "native route on all 56 rows; forced-action utility on clean plus all 11 severity-5 rows",
            "forward_count": {
                "native_route_forwards": native_forwards,
                "forced_action_forwards": forced_forwards,
                "total_model_forwards": total_forwards,
                "forced_actions_per_utility_row": len(actions),
            },
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        },
        "condition_sensitivity": sensitivity,
        "conditions": rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "limitations": [
            "This is deterministic single-seed descriptive evidence, not a multi-seed significance test.",
            "True utility uses per-sample task loss; it is not an AP delta and is not additive across samples.",
            "The best-subset oracle is restricted to the router's identity/single/Top-k action space.",
            "Two-stage mode has true forced-action utility only on clean and severity-5 rows; other rows are route-only.",
            "Subset/limit schedules are smoke diagnostics and are ineligible for manuscript tables.",
        ],
    }


def csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    names = report["router"]["expert_names"]
    rows = []
    for item in report["conditions"]:
        has_utility = bool(item["utility_evidence_available"])
        row = {
            "protocol": report["protocol"],
            "run_kind": report["run_kind"],
            "eligible_for_manuscript_table": report["eligible_for_manuscript_table"],
            "split": report["evaluation"]["split"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "checkpoint_seed": report["checkpoint"]["seed"],
            "condition_key": item["condition_key"],
            "condition_type": item["condition_type"],
            "condition": item["condition"],
            "severity": item["severity"],
            "parameters": json.dumps(item["parameters"], sort_keys=True),
            "corruption_seed": item["corruption_seed"],
            "samples": item["samples"],
            "utility_evidence_available": has_utility,
            "identity_fallback_rate": item["identity_fallback_rate"],
            "mean_active_experts": item["mean_active_experts"],
            "active_experts_std": item["active_experts_std"],
            "dominant_route_mask": item["dominant_route_mask"],
            "dominant_route_mask_rate": item["dominant_route_mask_rate"],
            "route_configuration_entropy_bits": item["route_configuration_entropy_bits"],
            "utility_spearman": item.get("utility_spearman"),
            "top1_single_oracle_match": item.get("top1_single_oracle_match"),
            "top2_single_oracle_coverage": item.get("top2_single_oracle_coverage"),
            "best_action_exact_match": item.get("best_action_exact_match"),
            "best_action_expert_recall": item.get("best_action_expert_recall"),
            "no_harm_rate": item.get("no_harm_rate"),
            "oracle_identity_rate": item.get("oracle_identity_rate"),
            "mean_routed_relative_utility": item.get("routed_relative_utility", {}).get("mean"),
            "mean_best_action_relative_utility": item.get("best_action_relative_utility", {}).get("mean"),
            "regret": item.get("regret"),
            "complete_action_regret": item.get("complete_action_regret"),
            "mean_regret_to_best_single": item.get("relative_utility_regret_to_best_single", {}).get("mean"),
            "mean_regret_to_best_action": item.get("relative_utility_regret_to_best_action", {}).get("mean"),
            "p95_regret_to_best_action": item.get("relative_utility_regret_to_best_action", {}).get("p95"),
            "route_forced_loss_max_abs_delta": item.get("route_forced_loss_max_abs_delta"),
            "top1_advantage_over_matched_random": item.get("reference_advantage", {}).get(
                "top1_over_matched_random"
            ),
            "top2_advantage_over_matched_random": item.get("reference_advantage", {}).get(
                "top2_over_matched_random"
            ),
            "best_action_exact_advantage_over_matched_random": item.get("reference_advantage", {}).get(
                "best_action_exact_over_matched_random"
            ),
            "no_harm_advantage_over_matched_random": item.get("reference_advantage", {}).get(
                "no_harm_over_matched_random"
            ),
            "regret_reduction_vs_matched_random": item.get("reference_advantage", {}).get(
                "regret_reduction_vs_matched_random"
            ),
            "regret_reduction_vs_zero_predictor": item.get("reference_advantage", {}).get(
                "regret_reduction_vs_zero_predictor"
            ),
            "activation_l1_from_clean": item["versus_clean"]["expert_activation_l1"],
            "route_mask_js_divergence_from_clean_bits": item["versus_clean"][
                "route_mask_js_divergence_bits"
            ],
            "identity_fallback_delta_from_clean": item["versus_clean"]["identity_fallback_rate_delta"],
            "mean_active_experts_delta_from_clean": item["versus_clean"]["mean_active_experts_delta"],
            "route_mask_distribution": json.dumps(item["route_mask_distribution"], sort_keys=True),
            "best_action_distribution": json.dumps(item.get("best_action_distribution", {}), sort_keys=True),
            "elapsed_seconds": item["elapsed_seconds"],
        }
        for name in names:
            row[f"activation_rate_{name}"] = item["expert_activation_rate"][name]
            row[f"predicted_utility_mean_{name}"] = item["predicted_utility_summary"][name]["mean"]
            row[f"predicted_utility_std_{name}"] = item["predicted_utility_summary"][name]["std"]
            row[f"predicted_utility_p05_{name}"] = item["predicted_utility_summary"][name]["p05"]
            row[f"predicted_utility_median_{name}"] = item["predicted_utility_summary"][name]["median"]
            row[f"predicted_utility_p95_{name}"] = item["predicted_utility_summary"][name]["p95"]
            true_summary = item.get("true_single_expert_utility_summary")
            row[f"true_utility_mean_{name}"] = true_summary[name]["mean"] if true_summary else None
            row[f"true_utility_std_{name}"] = true_summary[name]["std"] if true_summary else None
            row[f"utility_mae_{name}"] = item.get("utility_mae", {}).get(name)
            row[f"utility_bias_{name}"] = item.get("utility_bias", {}).get(name)
            row[f"selected_positive_precision_{name}"] = item.get("selected_positive_precision", {}).get(name)
        rows.append(row)
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
        description="Analyze UtilityRoute configuration and true utility across deterministic corruption conditions."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DATA_YAML,
        help="Paired DroneVehicle YOLO data YAML; defaults to the workspace config.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--severities", nargs="+", type=int, choices=SEVERITY_LEVELS, default=list(SEVERITY_LEVELS))
    parser.add_argument("--corruption-seed", type=int, default=42)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="full_matrix")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Dataset rectangular-batch geometry; inference remains sample-wise.")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    args.conditions = normalize_unique(args.conditions)
    args.severities = normalize_unique(args.severities)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.imgsz <= 0 or args.batch <= 0:
        parser.error("--imgsz and --batch must be positive")
    if not full_protocol_arguments(args) and "smoke" not in args.output_prefix.name.lower():
        parser.error("Partial schedules, non-default protocols, and --limit outputs must include 'smoke' in --output-prefix")
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
                "condition_rows": report["evaluation"]["condition_rows"],
                "samples_per_condition": report["evaluation"]["evaluated_samples_per_condition"],
                "configuration_varies": report["condition_sensitivity"]["configuration_varies_across_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
