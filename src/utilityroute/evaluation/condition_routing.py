"""Pure summaries for condition-wise UtilityRoute behavior and true utility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


IDENTITY = "identity"


def action_space(expert_names: Sequence[str], top_k: int) -> dict[str, tuple[int, ...]]:
    """Return every identity/sparse mask available to a Top-k router."""

    names = tuple(expert_names)
    if not names:
        raise ValueError("expert_names must not be empty")
    if not 1 <= top_k <= len(names):
        raise ValueError("top_k must be between one and the number of experts")
    actions: dict[str, tuple[int, ...]] = {IDENTITY: (0,) * len(names)}
    for active_count in range(1, top_k + 1):
        for indices in combinations(range(len(names)), active_count):
            mask = tuple(int(index in indices) for index in range(len(names)))
            label = "+".join(names[index] for index in indices)
            actions[label] = mask
    return actions


def mask_label(mask: Sequence[int | bool], expert_names: Sequence[str]) -> str:
    values = tuple(bool(value) for value in mask)
    if len(values) != len(expert_names):
        raise ValueError("mask length must match expert_names")
    selected = [name for name, active in zip(expert_names, values) if active]
    return "+".join(selected) if selected else IDENTITY


@dataclass(frozen=True)
class NativeRouteObservation:
    """One sample's native predicted utilities, scores, and active mask."""

    predicted_utility: np.ndarray
    net_score: np.ndarray
    active_mask: np.ndarray

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_utility, dtype=np.float64).reshape(-1)
        score = np.asarray(self.net_score, dtype=np.float64).reshape(-1)
        mask = np.asarray(self.active_mask, dtype=bool).reshape(-1)
        if not len(predicted) or len(predicted) != len(score) or len(predicted) != len(mask):
            raise ValueError("predicted_utility, net_score, and active_mask must have equal non-zero length")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(score)):
            raise ValueError("predicted utilities and net scores must be finite")
        object.__setattr__(self, "predicted_utility", predicted)
        object.__setattr__(self, "net_score", score)
        object.__setattr__(self, "active_mask", mask)


@dataclass(frozen=True)
class RoutingObservation:
    """One sample's native route and losses for every allowed forced action."""

    predicted_utility: np.ndarray
    net_score: np.ndarray
    active_mask: np.ndarray
    identity_loss: float
    routed_loss: float
    action_losses: Mapping[str, float]

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_utility, dtype=np.float64).reshape(-1)
        score = np.asarray(self.net_score, dtype=np.float64).reshape(-1)
        mask = np.asarray(self.active_mask, dtype=bool).reshape(-1)
        if not len(predicted) or len(predicted) != len(score) or len(predicted) != len(mask):
            raise ValueError("predicted_utility, net_score, and active_mask must have equal non-zero length")
        losses = {str(key): float(value) for key, value in self.action_losses.items()}
        if IDENTITY not in losses:
            raise ValueError("action_losses must contain identity")
        if self.identity_loss < 0 or self.routed_loss < 0 or any(value < 0 for value in losses.values()):
            raise ValueError("task losses must be nonnegative")
        if not all(np.isfinite(value) for value in [self.identity_loss, self.routed_loss, *losses.values()]):
            raise ValueError("loss values must be finite")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(score)):
            raise ValueError("predicted utilities and net scores must be finite")
        object.__setattr__(self, "predicted_utility", predicted)
        object.__setattr__(self, "net_score", score)
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(self, "identity_loss", float(self.identity_loss))
        object.__setattr__(self, "routed_loss", float(self.routed_loss))
        object.__setattr__(self, "action_losses", losses)


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        raise ValueError("distribution values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0)),
    }


def summarize_native_routes(
    observations: Sequence[NativeRouteObservation], expert_names: Sequence[str]
) -> dict[str, object]:
    """Summarize route activation and predicted utility without forced actions."""

    rows = list(observations)
    names = tuple(expert_names)
    if not rows:
        raise ValueError("observations must not be empty")
    if any(len(row.predicted_utility) != len(names) for row in rows):
        raise ValueError("observation width must match expert_names")
    activations = np.zeros(len(names), dtype=np.int64)
    histogram = {str(index): 0 for index in range(len(names) + 1)}
    mask_counts: dict[str, int] = {}
    predicted_values = [[] for _ in names]
    active_counts = []
    for row in rows:
        count = int(row.active_mask.sum())
        active_counts.append(count)
        histogram[str(count)] += 1
        activations += row.active_mask.astype(np.int64)
        label = mask_label(row.active_mask, names)
        mask_counts[label] = mask_counts.get(label, 0) + 1
        for index, value in enumerate(row.predicted_utility):
            predicted_values[index].append(float(value))
    samples = len(rows)
    distribution = {label: count / samples for label, count in mask_counts.items()}
    dominant = max(distribution, key=distribution.get)
    entropy = -sum(value * math.log2(value) for value in distribution.values() if value > 0)
    return {
        "samples": samples,
        "identity_fallback_rate": histogram["0"] / samples,
        "mean_active_experts": float(np.mean(active_counts)),
        "active_experts_std": float(np.std(active_counts)),
        "active_expert_histogram": histogram,
        "expert_activation_rate": {
            name: int(activations[index]) / samples for index, name in enumerate(names)
        },
        "route_mask_distribution": distribution,
        "dominant_route_mask": dominant,
        "dominant_route_mask_rate": distribution[dominant],
        "route_configuration_entropy_bits": entropy,
        "predicted_utility_summary": {
            name: distribution_summary(predicted_values[index]) for index, name in enumerate(names)
        },
    }


def _best_action(losses: Mapping[str, float], ordered_actions: Sequence[str]) -> str:
    """Pick minimum loss, preferring the earlier (sparser) action on exact ties."""

    return min(ordered_actions, key=lambda label: (float(losses[label]), ordered_actions.index(label)))


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return None if math.isnan(value) else value


def summarize_condition(
    observations: Sequence[RoutingObservation],
    expert_names: Sequence[str],
    actions: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Summarize configuration behavior and on-condition forced-action utility."""

    rows = list(observations)
    names = tuple(expert_names)
    action_masks = {label: tuple(int(value) for value in mask) for label, mask in actions.items()}
    if not rows:
        raise ValueError("observations must not be empty")
    if list(action_masks)[0] != IDENTITY or action_masks[IDENTITY] != (0,) * len(names):
        raise ValueError("actions must start with an all-zero identity mask")
    if any(len(mask) != len(names) for mask in action_masks.values()):
        raise ValueError("every action mask must match expert_names")
    if any(set(row.action_losses) != set(action_masks) for row in rows):
        raise ValueError("every observation must provide exactly the declared action losses")
    if any(len(row.predicted_utility) != len(names) for row in rows):
        raise ValueError("observation width must match expert_names")

    ordered_actions = list(action_masks)
    single_actions = [IDENTITY, *names]
    if any(label not in action_masks for label in single_actions):
        raise ValueError("actions must include identity and all single experts")

    native_summary = summarize_native_routes(
        [
            NativeRouteObservation(row.predicted_utility, row.net_score, row.active_mask)
            for row in rows
        ],
        names,
    )
    activations = np.zeros(len(names), dtype=np.int64)
    selected_positive = np.zeros(len(names), dtype=np.int64)
    predicted_values = [[] for _ in names]
    true_single_values = [[] for _ in names]
    action_utilities = {label: [] for label in ordered_actions}
    route_masks = {label: 0 for label in ordered_actions}
    best_actions = {label: 0 for label in ordered_actions}
    best_single_actions = {label: 0 for label in single_actions}
    active_histogram = {str(index): 0 for index in range(len(names) + 1)}
    routed_utilities: list[float] = []
    best_action_utilities: list[float] = []
    best_single_utilities: list[float] = []
    regrets: list[float] = []
    single_regrets: list[float] = []
    forced_route_deltas: list[float] = []
    no_harm = top1_match = top2_coverage = exact_action = 0
    random_top1 = random_top2 = random_exact = random_no_harm = random_regret = 0.0
    oracle_overlap: list[float] = []
    identity_losses: list[float] = []
    routed_losses: list[float] = []

    for row in rows:
        identity = row.identity_loss
        denominator = identity + 1e-6
        identity_losses.append(identity)
        routed_losses.append(row.routed_loss)
        mask = row.active_mask.astype(bool)
        active_count = int(mask.sum())
        active_histogram[str(active_count)] += 1
        activations += mask.astype(np.int64)
        route_label = mask_label(mask, names)
        if route_label not in action_masks:
            raise ValueError(f"Native route {route_label!r} is outside the declared action space")
        route_masks[route_label] += 1
        forced_route_deltas.append(abs(row.routed_loss - row.action_losses[route_label]))

        utilities = {
            label: (identity - float(row.action_losses[label])) / denominator for label in ordered_actions
        }
        for label, value in utilities.items():
            action_utilities[label].append(value)
        routed_utility = (identity - row.routed_loss) / denominator
        routed_utilities.append(routed_utility)

        for index, name in enumerate(names):
            prediction = float(row.predicted_utility[index])
            true_utility = utilities[name]
            predicted_values[index].append(prediction)
            true_single_values[index].append(true_utility)
            selected_positive[index] += int(mask[index] and true_utility > 0)

        best_label = _best_action(row.action_losses, ordered_actions)
        best_single_label = _best_action(row.action_losses, single_actions)
        best_actions[best_label] += 1
        best_single_actions[best_single_label] += 1
        best_utility = utilities[best_label]
        best_single_utility = utilities[best_single_label]
        best_action_utilities.append(best_utility)
        best_single_utilities.append(best_single_utility)
        regrets.append(best_utility - routed_utility)
        single_regrets.append(best_single_utility - routed_utility)
        no_harm += int(row.routed_loss <= identity)
        exact_action += int(route_label == best_label)

        predicted_top1 = IDENTITY if np.max(row.net_score) <= 0 else names[int(np.argmax(row.net_score))]
        top1_match += int(predicted_top1 == best_single_label)
        top2_coverage += int(
            (best_single_label == IDENTITY and active_count == 0)
            or (best_single_label != IDENTITY and bool(mask[names.index(best_single_label)]))
        )
        oracle_mask = np.asarray(action_masks[best_label], dtype=bool)
        if not oracle_mask.any():
            oracle_overlap.append(float(active_count == 0))
        else:
            oracle_overlap.append(float(np.logical_and(mask, oracle_mask).sum() / oracle_mask.sum()))

        if active_count == 0:
            random_labels = [IDENTITY]
            random_top1 += float(best_single_label == IDENTITY)
            random_top2 += float(best_single_label == IDENTITY)
        else:
            random_labels = [
                label for label, candidate_mask in action_masks.items() if sum(candidate_mask) == active_count
            ]
            random_top1 += 0.0 if best_single_label == IDENTITY else 1.0 / len(names)
            random_top2 += 0.0 if best_single_label == IDENTITY else active_count / len(names)
        random_exact += sum(label == best_label for label in random_labels) / len(random_labels)
        random_no_harm += sum(row.action_losses[label] <= identity for label in random_labels) / len(
            random_labels
        )
        random_regret += sum(best_utility - utilities[label] for label in random_labels) / len(random_labels)

    samples = len(rows)
    route_distribution = {label: count / samples for label, count in route_masks.items() if count}
    best_distribution = {label: count / samples for label, count in best_actions.items() if count}
    best_single_distribution = {label: count / samples for label, count in best_single_actions.items() if count}
    route_entropy = -sum(value * math.log2(value) for value in route_distribution.values() if value > 0)
    dominant_route = max(route_distribution, key=route_distribution.get)

    predicted_summary = {
        name: distribution_summary(predicted_values[index]) for index, name in enumerate(names)
    }
    true_summary = {
        name: distribution_summary(true_single_values[index]) for index, name in enumerate(names)
    }
    utility_mae = {
        name: float(np.mean(np.abs(np.asarray(predicted_values[index]) - np.asarray(true_single_values[index]))))
        for index, name in enumerate(names)
    }
    utility_bias = {
        name: float(np.mean(np.asarray(predicted_values[index]) - np.asarray(true_single_values[index])))
        for index, name in enumerate(names)
    }
    flat_predictions = [value for expert in predicted_values for value in expert]
    flat_targets = [value for expert in true_single_values for value in expert]
    active_counts = [int(row.active_mask.sum()) for row in rows]
    regret_summary = distribution_summary(regrets)
    matched_random = {
        "top1_single_oracle_match_expected": random_top1 / samples,
        "top2_single_oracle_coverage_expected": random_top2 / samples,
        "best_action_exact_match_expected": random_exact / samples,
        "no_harm_rate_expected": random_no_harm / samples,
        "mean_relative_utility_regret_to_best_action_expected": random_regret / samples,
        "mean_relative_utility_regret_to_best_single_expected": float(np.mean(best_single_utilities)) - float(np.mean(best_action_utilities)) + random_regret / samples,
        "mean_active_experts": float(np.mean(active_counts)),
        "definition": "uniform random expert mask with the native route's per-sample active-expert count; fallback stays identity",
    }
    zero_predictor = {
        "top1_single_oracle_match": best_single_actions[IDENTITY] / samples,
        "top2_single_oracle_coverage": best_single_actions[IDENTITY] / samples,
        "best_action_exact_match": best_actions[IDENTITY] / samples,
        "no_harm_rate": 1.0,
        "mean_relative_utility_regret_to_best_action": float(np.mean(best_action_utilities)),
        "mean_relative_utility_regret_to_best_single": float(np.mean(best_single_utilities)),
        "utility_mae": {
            name: float(np.mean(np.abs(true_single_values[index]))) for index, name in enumerate(names)
        },
        "utility_spearman": None,
        "definition": "predict zero utility and always retain Identity",
    }
    reference_advantage = {
        "top1_over_matched_random": top1_match / samples - matched_random["top1_single_oracle_match_expected"],
        "top2_over_matched_random": top2_coverage / samples
        - matched_random["top2_single_oracle_coverage_expected"],
        "best_action_exact_over_matched_random": exact_action / samples
        - matched_random["best_action_exact_match_expected"],
        "no_harm_over_matched_random": no_harm / samples - matched_random["no_harm_rate_expected"],
        "regret_reduction_vs_matched_random": matched_random[
            "mean_relative_utility_regret_to_best_action_expected"
        ]
        - regret_summary["mean"],
        "regret_reduction_vs_zero_predictor": zero_predictor[
            "mean_relative_utility_regret_to_best_action"
        ]
        - regret_summary["mean"],
    }

    return {
        **native_summary,
        "true_single_expert_utility_summary": true_summary,
        "true_action_utility_summary": {
            label: distribution_summary(values) for label, values in action_utilities.items()
        },
        "utility_mae": utility_mae,
        "utility_bias": utility_bias,
        "utility_spearman": _safe_spearman(flat_predictions, flat_targets),
        "selected_positive_precision": {
            name: (int(selected_positive[index]) / int(activations[index]) if activations[index] else None)
            for index, name in enumerate(names)
        },
        "top1_single_oracle_match": top1_match / samples,
        "top2_single_oracle_coverage": top2_coverage / samples,
        "best_action_exact_match": exact_action / samples,
        "best_action_expert_recall": float(np.mean(oracle_overlap)),
        "no_harm_rate": no_harm / samples,
        "oracle_identity_rate": best_actions[IDENTITY] / samples,
        "single_oracle_identity_rate": best_single_actions[IDENTITY] / samples,
        "best_action_distribution": best_distribution,
        "best_single_action_distribution": best_single_distribution,
        "routed_relative_utility": distribution_summary(routed_utilities),
        "best_action_relative_utility": distribution_summary(best_action_utilities),
        "best_single_relative_utility": distribution_summary(best_single_utilities),
        "relative_utility_regret_to_best_action": regret_summary,
        "relative_utility_regret_to_best_single": distribution_summary(single_regrets),
        "regret": float(np.mean(single_regrets)),
        "complete_action_regret": float(np.mean(regrets)),
        "metric_definition": {
            "utility": "(L_identity - L_action) / (L_identity + 1e-6)",
            "regret": "max(0, max single-expert utility) - routed utility",
            "complete_action_regret": "max utility over all declared actions - routed utility",
            "scale": "rates and utilities use unit scale, not percentages",
        },
        "identity_loss": distribution_summary(identity_losses),
        "routed_loss": distribution_summary(routed_losses),
        "route_forced_loss_max_abs_delta": max(forced_route_deltas),
        "route_forced_loss_mean_abs_delta": float(np.mean(forced_route_deltas)),
        "zero_predictor_reference": zero_predictor,
        "matched_random_reference": matched_random,
        "reference_advantage": reference_advantage,
    }


def jensen_shannon_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Base-2 Jensen-Shannon divergence over sparse categorical mappings."""

    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    q = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("categorical distributions must have positive mass")
    p /= p.sum()
    q /= q.sum()
    midpoint = (p + q) / 2.0

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * (kl(p) + kl(q))


def add_clean_comparisons(rows: Sequence[dict[str, object]], expert_names: Sequence[str]) -> None:
    """Attach descriptive route/utility shifts relative to the single clean row."""

    clean_rows = [row for row in rows if row.get("condition") == "clean"]
    if len(clean_rows) != 1:
        raise ValueError("exactly one clean row is required for clean comparisons")
    clean = clean_rows[0]
    clean_activation = clean["expert_activation_rate"]
    clean_prediction = clean["predicted_utility_summary"]
    clean_truth = clean.get("true_single_expert_utility_summary")
    clean_masks = clean["route_mask_distribution"]
    for row in rows:
        activation_delta = {
            name: float(row["expert_activation_rate"][name]) - float(clean_activation[name])
            for name in expert_names
        }
        prediction_delta = {
            name: float(row["predicted_utility_summary"][name]["mean"])
            - float(clean_prediction[name]["mean"])
            for name in expert_names
        }
        row_truth = row.get("true_single_expert_utility_summary")
        truth_delta = None
        if row_truth is not None and clean_truth is not None:
            truth_delta = {
                name: float(row_truth[name]["mean"]) - float(clean_truth[name]["mean"])
                for name in expert_names
            }
        route_jsd = jensen_shannon_divergence(row["route_mask_distribution"], clean_masks)
        row["versus_clean"] = {
            "expert_activation_rate_delta": activation_delta,
            "expert_activation_l1": float(sum(abs(value) for value in activation_delta.values())),
            "expert_activation_max_abs": float(max(abs(value) for value in activation_delta.values())),
            "identity_fallback_rate_delta": float(row["identity_fallback_rate"])
            - float(clean["identity_fallback_rate"]),
            "mean_active_experts_delta": float(row["mean_active_experts"])
            - float(clean["mean_active_experts"]),
            "route_mask_js_divergence_bits": route_jsd,
            "predicted_utility_mean_delta": prediction_delta,
            "true_single_utility_mean_delta": truth_delta,
            "configuration_changed_descriptively": bool(
                route_jsd > 1e-12 or any(abs(value) > 1e-12 for value in activation_delta.values())
            ),
        }


def condition_sensitivity_summary(
    rows: Sequence[dict[str, object]], expert_names: Sequence[str]
) -> dict[str, object]:
    """Summarize whether route configurations vary across condition/severity rows."""

    if not rows:
        raise ValueError("rows must not be empty")
    ranges = {}
    predicted_mean_ranges = {}
    true_mean_ranges = {}
    for name in expert_names:
        values = [float(row["expert_activation_rate"][name]) for row in rows]
        ranges[name] = {"min": min(values), "max": max(values), "range": max(values) - min(values)}
        predicted_values = [float(row["predicted_utility_summary"][name]["mean"]) for row in rows]
        true_values = [
            float(row["true_single_expert_utility_summary"][name]["mean"])
            for row in rows
            if row.get("true_single_expert_utility_summary") is not None
        ]
        predicted_mean_ranges[name] = {
            "min": min(predicted_values),
            "max": max(predicted_values),
            "range": max(predicted_values) - min(predicted_values),
        }
        true_mean_ranges[name] = (
            {"min": min(true_values), "max": max(true_values), "range": max(true_values) - min(true_values)}
            if true_values
            else None
        )
    dominant_masks = sorted({str(row["dominant_route_mask"]) for row in rows})
    clean_shifts = [
        (float(row["versus_clean"]["route_mask_js_divergence_bits"]), str(row["condition_key"]))
        for row in rows
    ]
    max_jsd, max_jsd_condition = max(clean_shifts)

    severity_trends: dict[str, object] = {}
    degraded_conditions = sorted({str(row["condition"]) for row in rows if row["condition"] != "clean"})
    for condition in degraded_conditions:
        condition_rows = sorted(
            (row for row in rows if row["condition"] == condition), key=lambda row: int(row["severity"])
        )
        severity = [int(row["severity"]) for row in condition_rows]
        utility_rows = [row for row in condition_rows if row.get("utility_evidence_available", True)]
        utility_severity = [int(row["severity"]) for row in utility_rows]
        severity_trends[condition] = {
            "rows": len(condition_rows),
            "identity_fallback_spearman": _safe_spearman(
                severity, [float(row["identity_fallback_rate"]) for row in condition_rows]
            ),
            "mean_active_experts_spearman": _safe_spearman(
                severity, [float(row["mean_active_experts"]) for row in condition_rows]
            ),
            "route_js_divergence_spearman": _safe_spearman(
                severity,
                [float(row["versus_clean"]["route_mask_js_divergence_bits"]) for row in condition_rows],
            ),
            "no_harm_spearman": _safe_spearman(
                utility_severity, [float(row["no_harm_rate"]) for row in utility_rows]
            ),
            "regret_spearman": _safe_spearman(
                utility_severity,
                [float(row["relative_utility_regret_to_best_action"]["mean"]) for row in utility_rows],
            ),
            "expert_activation_spearman": {
                name: _safe_spearman(
                    severity, [float(row["expert_activation_rate"][name]) for row in condition_rows]
                )
                for name in expert_names
            },
            "predicted_utility_mean_spearman": {
                name: _safe_spearman(
                    severity,
                    [float(row["predicted_utility_summary"][name]["mean"]) for row in condition_rows],
                )
                for name in expert_names
            },
            "true_single_utility_mean_spearman": {
                name: _safe_spearman(
                    utility_severity,
                    [float(row["true_single_expert_utility_summary"][name]["mean"]) for row in utility_rows],
                )
                for name in expert_names
            },
        }
    return {
        "rows": len(rows),
        "distinct_dominant_route_masks": dominant_masks,
        "distinct_dominant_route_mask_count": len(dominant_masks),
        "expert_activation_ranges": ranges,
        "predicted_utility_mean_ranges": predicted_mean_ranges,
        "true_single_utility_mean_ranges": true_mean_ranges,
        "max_route_js_divergence_from_clean_bits": max_jsd,
        "max_route_js_divergence_condition": max_jsd_condition,
        "rows_with_descriptive_configuration_change": sum(
            bool(row["versus_clean"]["configuration_changed_descriptively"]) for row in rows
        ),
        "configuration_varies_across_rows": bool(
            len(dominant_masks) > 1 or any(value["range"] > 1e-12 for value in ranges.values())
        ),
        "predicted_utility_means_vary_across_rows": any(
            value["range"] > 1e-12 for value in predicted_mean_ranges.values()
        ),
        "true_single_utility_means_vary_across_rows": any(
            value is not None and value["range"] > 1e-12 for value in true_mean_ranges.values()
        ),
        "utility_evidence_rows": sum(bool(row.get("utility_evidence_available", True)) for row in rows),
        "severity_trends": severity_trends,
        "interpretation_note": (
            "These are descriptive condition-wise shifts. Statistical generalization requires repeated seeds or "
            "sample-level uncertainty analysis."
        ),
    }
