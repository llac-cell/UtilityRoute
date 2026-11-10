"""Small, dependency-light MOT metrics compatible with TrackEval definitions.

The CLEAR, Identity, and HOTA calculations below follow the sequence-level
definitions in the official TrackEval reference implementation. Keeping the
core arithmetic here makes the reported numbers testable without installing a
second evaluation framework or mutating the experiment environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


HOTA_THRESHOLDS = np.arange(0.05, 0.99, 0.05, dtype=np.float64)


@dataclass(frozen=True)
class TrackingFrame:
    """Ground truth, detections, and tracker output for one video frame."""

    gt_ids: np.ndarray
    gt_boxes: np.ndarray
    pred_ids: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray

    def __post_init__(self) -> None:
        gt_ids = np.asarray(self.gt_ids, dtype=np.int64).reshape(-1)
        pred_ids = np.asarray(self.pred_ids, dtype=np.int64).reshape(-1)
        gt_boxes = np.asarray(self.gt_boxes, dtype=np.float64).reshape(-1, 4)
        pred_boxes = np.asarray(self.pred_boxes, dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(self.pred_scores, dtype=np.float64).reshape(-1)
        if len(gt_ids) != len(gt_boxes):
            raise ValueError("gt_ids and gt_boxes must have equal length")
        if len(pred_ids) != len(pred_boxes) or len(pred_ids) != len(pred_scores):
            raise ValueError("pred_ids, pred_boxes, and pred_scores must have equal length")
        if len(set(gt_ids.tolist())) != len(gt_ids) or len(set(pred_ids.tolist())) != len(pred_ids):
            raise ValueError("track IDs must be unique within each frame")
        if not np.all(np.isfinite(gt_boxes)) or not np.all(np.isfinite(pred_boxes)) or not np.all(np.isfinite(pred_scores)):
            raise ValueError("boxes and scores must be finite")
        if np.any(gt_boxes[:, 2:] <= gt_boxes[:, :2]) or np.any(pred_boxes[:, 2:] <= pred_boxes[:, :2]):
            raise ValueError("boxes must have positive width and height")
        object.__setattr__(self, "gt_ids", gt_ids)
        object.__setattr__(self, "gt_boxes", gt_boxes)
        object.__setattr__(self, "pred_ids", pred_ids)
        object.__setattr__(self, "pred_boxes", pred_boxes)
        object.__setattr__(self, "pred_scores", pred_scores)


def box_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for axis-aligned ``xyxy`` boxes."""

    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _canonical_ids(frames: Sequence[TrackingFrame]) -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    gt_values = sorted({int(value) for frame in frames for value in frame.gt_ids})
    pred_values = sorted({int(value) for frame in frames for value in frame.pred_ids})
    gt_map = {value: index for index, value in enumerate(gt_values)}
    pred_map = {value: index for index, value in enumerate(pred_values)}
    gt_ids = [np.asarray([gt_map[int(value)] for value in frame.gt_ids], dtype=np.int64) for frame in frames]
    pred_ids = [np.asarray([pred_map[int(value)] for value in frame.pred_ids], dtype=np.int64) for frame in frames]
    return gt_ids, pred_ids, len(gt_values), len(pred_values)


def _similarities(frames: Sequence[TrackingFrame]) -> list[np.ndarray]:
    return [box_iou_xyxy(frame.gt_boxes, frame.pred_boxes) for frame in frames]


def _clear_metrics(
    gt_ids: Sequence[np.ndarray],
    pred_ids: Sequence[np.ndarray],
    similarities: Sequence[np.ndarray],
    num_gt_ids: int,
    threshold: float,
) -> dict[str, float | int]:
    previous_tracker = np.full(num_gt_ids, np.nan)
    previous_timestep_tracker = np.full(num_gt_ids, np.nan)
    true_positives = false_negatives = false_positives = id_switches = 0
    motp_sum = 0.0

    for gt_ids_t, pred_ids_t, similarity in zip(gt_ids, pred_ids, similarities):
        if not len(gt_ids_t):
            false_positives += len(pred_ids_t)
            continue
        if not len(pred_ids_t):
            false_negatives += len(gt_ids_t)
            continue
        continuation = pred_ids_t[None, :] == previous_timestep_tracker[gt_ids_t[:, None]]
        score = continuation.astype(np.float64) * 1000.0 + similarity
        score[similarity < threshold - np.finfo(float).eps] = 0.0
        rows, cols = linear_sum_assignment(-score)
        valid = score[rows, cols] > np.finfo(float).eps
        rows, cols = rows[valid], cols[valid]
        matched_gt = gt_ids_t[rows]
        matched_pred = pred_ids_t[cols]
        old_pred = previous_tracker[matched_gt]
        id_switches += int(np.sum((~np.isnan(old_pred)) & (matched_pred != old_pred)))
        previous_tracker[matched_gt] = matched_pred
        previous_timestep_tracker[:] = np.nan
        previous_timestep_tracker[matched_gt] = matched_pred
        true_positives += len(rows)
        false_negatives += len(gt_ids_t) - len(rows)
        false_positives += len(pred_ids_t) - len(rows)
        motp_sum += float(np.sum(similarity[rows, cols]))

    gt_detections = true_positives + false_negatives
    mota = (true_positives - false_positives - id_switches) / max(1.0, gt_detections)
    return {
        "MOTA": float(mota),
        "MOTP": float(motp_sum / max(1.0, true_positives)),
        "CLR_TP": int(true_positives),
        "CLR_FN": int(false_negatives),
        "CLR_FP": int(false_positives),
        "IDSW": int(id_switches),
        "MOTP_sum": float(motp_sum),
        "frames": int(len(gt_ids)),
    }


def _identity_metrics(
    gt_ids: Sequence[np.ndarray],
    pred_ids: Sequence[np.ndarray],
    similarities: Sequence[np.ndarray],
    num_gt_ids: int,
    num_pred_ids: int,
    threshold: float,
) -> dict[str, float | int]:
    if not num_pred_ids:
        total_gt = sum(len(ids) for ids in gt_ids)
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0, "IDTP": 0, "IDFN": total_gt, "IDFP": 0}
    if not num_gt_ids:
        total_pred = sum(len(ids) for ids in pred_ids)
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0, "IDTP": 0, "IDFN": 0, "IDFP": total_pred}

    potential = np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)
    gt_count = np.zeros(num_gt_ids, dtype=np.float64)
    pred_count = np.zeros(num_pred_ids, dtype=np.float64)
    for gt_ids_t, pred_ids_t, similarity in zip(gt_ids, pred_ids, similarities):
        match_gt, match_pred = np.nonzero(similarity >= threshold - np.finfo(float).eps)
        potential[gt_ids_t[match_gt], pred_ids_t[match_pred]] += 1
        gt_count[gt_ids_t] += 1
        pred_count[pred_ids_t] += 1

    size = num_gt_ids + num_pred_ids
    false_positive_cost = np.zeros((size, size), dtype=np.float64)
    false_negative_cost = np.zeros((size, size), dtype=np.float64)
    false_positive_cost[num_gt_ids:, :num_pred_ids] = 1e10
    false_negative_cost[:num_gt_ids, num_pred_ids:] = 1e10
    for gt_id in range(num_gt_ids):
        false_negative_cost[gt_id, :num_pred_ids] = gt_count[gt_id]
        false_negative_cost[gt_id, num_pred_ids + gt_id] = gt_count[gt_id]
    for pred_id in range(num_pred_ids):
        false_positive_cost[:num_gt_ids, pred_id] = pred_count[pred_id]
        false_positive_cost[num_gt_ids + pred_id, pred_id] = pred_count[pred_id]
    false_negative_cost[:num_gt_ids, :num_pred_ids] -= potential
    false_positive_cost[:num_gt_ids, :num_pred_ids] -= potential
    rows, cols = linear_sum_assignment(false_negative_cost + false_positive_cost)
    identity_fn = int(round(float(false_negative_cost[rows, cols].sum())))
    identity_fp = int(round(float(false_positive_cost[rows, cols].sum())))
    identity_tp = int(round(float(gt_count.sum()))) - identity_fn
    identity_recall = identity_tp / max(1.0, identity_tp + identity_fn)
    identity_precision = identity_tp / max(1.0, identity_tp + identity_fp)
    identity_f1 = identity_tp / max(1.0, identity_tp + 0.5 * identity_fp + 0.5 * identity_fn)
    return {
        "IDF1": float(identity_f1),
        "IDP": float(identity_precision),
        "IDR": float(identity_recall),
        "IDTP": int(identity_tp),
        "IDFN": int(identity_fn),
        "IDFP": int(identity_fp),
    }


def _hota_metrics(
    gt_ids: Sequence[np.ndarray],
    pred_ids: Sequence[np.ndarray],
    similarities: Sequence[np.ndarray],
    num_gt_ids: int,
    num_pred_ids: int,
    thresholds: np.ndarray,
) -> dict[str, object]:
    count = len(thresholds)
    hota_tp = np.zeros(count, dtype=np.float64)
    hota_fn = np.zeros(count, dtype=np.float64)
    hota_fp = np.zeros(count, dtype=np.float64)
    location_sum = np.zeros(count, dtype=np.float64)
    potential = np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)
    gt_count = np.zeros((num_gt_ids, 1), dtype=np.float64)
    pred_count = np.zeros((1, num_pred_ids), dtype=np.float64)

    for gt_ids_t, pred_ids_t, similarity in zip(gt_ids, pred_ids, similarities):
        denominator = similarity.sum(axis=0)[None, :] + similarity.sum(axis=1)[:, None] - similarity
        normalized = np.divide(similarity, denominator, out=np.zeros_like(similarity), where=denominator > np.finfo(float).eps)
        if len(gt_ids_t) and len(pred_ids_t):
            potential[gt_ids_t[:, None], pred_ids_t[None, :]] += normalized
        gt_count[gt_ids_t] += 1
        pred_count[0, pred_ids_t] += 1
    denominator = gt_count + pred_count - potential
    global_alignment = np.divide(potential, denominator, out=np.zeros_like(potential), where=denominator > 0)
    match_counts = [np.zeros_like(potential) for _ in thresholds]

    for gt_ids_t, pred_ids_t, similarity in zip(gt_ids, pred_ids, similarities):
        if not len(gt_ids_t):
            hota_fp += len(pred_ids_t)
            continue
        if not len(pred_ids_t):
            hota_fn += len(gt_ids_t)
            continue
        score = global_alignment[gt_ids_t[:, None], pred_ids_t[None, :]] * similarity
        rows, cols = linear_sum_assignment(-score)
        for index, alpha in enumerate(thresholds):
            valid = similarity[rows, cols] >= alpha - np.finfo(float).eps
            selected_rows, selected_cols = rows[valid], cols[valid]
            matches = len(selected_rows)
            hota_tp[index] += matches
            hota_fn[index] += len(gt_ids_t) - matches
            hota_fp[index] += len(pred_ids_t) - matches
            if matches:
                location_sum[index] += float(similarity[selected_rows, selected_cols].sum())
                match_counts[index][gt_ids_t[selected_rows], pred_ids_t[selected_cols]] += 1

    association = np.zeros(count, dtype=np.float64)
    for index, matches in enumerate(match_counts):
        association_accuracy = matches / np.maximum(1.0, gt_count + pred_count - matches)
        association[index] = np.sum(matches * association_accuracy) / max(1.0, hota_tp[index])
    detection = hota_tp / np.maximum(1.0, hota_tp + hota_fn + hota_fp)
    hota = np.sqrt(detection * association)
    localization = np.maximum(1e-10, location_sum) / np.maximum(1e-10, hota_tp)
    return {
        "HOTA": float(np.mean(hota)),
        "DetA": float(np.mean(detection)),
        "AssA": float(np.mean(association)),
        "LocA": float(np.mean(localization)),
        "thresholds": [float(value) for value in thresholds],
        "HOTA_by_threshold": [float(value) for value in hota],
        "DetA_by_threshold": [float(value) for value in detection],
        "AssA_by_threshold": [float(value) for value in association],
        "LocA_by_threshold": [float(value) for value in localization],
        "HOTA_TP": [int(value) for value in hota_tp],
        "HOTA_FN": [int(value) for value in hota_fn],
        "HOTA_FP": [int(value) for value in hota_fp],
    }


def tracking_metrics(
    frames: Sequence[TrackingFrame],
    *,
    threshold: float = 0.5,
    hota_thresholds: np.ndarray = HOTA_THRESHOLDS,
) -> dict[str, object]:
    """Compute TrackEval-compatible CLEAR MOTA, Identity IDF1, and HOTA."""

    if not 0 < threshold <= 1:
        raise ValueError("matching threshold must be in (0, 1]")
    thresholds = np.asarray(hota_thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or not len(thresholds) or not np.all(np.isfinite(thresholds)) or np.any((thresholds <= 0) | (thresholds > 1)):
        raise ValueError("HOTA thresholds must be finite nonempty values in (0, 1]")
    frame_list = list(frames)
    gt_ids, pred_ids, num_gt_ids, num_pred_ids = _canonical_ids(frame_list)
    similarities = _similarities(frame_list)
    clear = _clear_metrics(gt_ids, pred_ids, similarities, num_gt_ids, threshold)
    identity = _identity_metrics(gt_ids, pred_ids, similarities, num_gt_ids, num_pred_ids, threshold)
    hota = _hota_metrics(
        gt_ids,
        pred_ids,
        similarities,
        num_gt_ids,
        num_pred_ids,
        thresholds,
    )
    return {
        "MOTA": clear["MOTA"],
        "IDF1": identity["IDF1"],
        "HOTA": hota["HOTA"],
        "clear_details": clear,
        "identity_details": identity,
        "hota_details": hota,
    }


def combine_sequence_metrics(results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Combine sequence metrics using TrackEval's detection-weighted rules."""

    if not results:
        return tracking_metrics([])
    if any(item["hota_details"]["thresholds"] != results[0]["hota_details"]["thresholds"] for item in results):
        raise ValueError("all sequences must use the same HOTA thresholds")
    clear_items = [item["clear_details"] for item in results]
    identity_items = [item["identity_details"] for item in results]
    hota_items = [item["hota_details"] for item in results]

    clear_tp = sum(int(item["CLR_TP"]) for item in clear_items)
    clear_fn = sum(int(item["CLR_FN"]) for item in clear_items)
    clear_fp = sum(int(item["CLR_FP"]) for item in clear_items)
    id_switches = sum(int(item["IDSW"]) for item in clear_items)
    motp_sum = sum(float(item["MOTP_sum"]) for item in clear_items)
    clear = {
        "MOTA": float((clear_tp - clear_fp - id_switches) / max(1.0, clear_tp + clear_fn)),
        "MOTP": float(motp_sum / max(1.0, clear_tp)),
        "CLR_TP": clear_tp,
        "CLR_FN": clear_fn,
        "CLR_FP": clear_fp,
        "IDSW": id_switches,
        "MOTP_sum": float(motp_sum),
        "frames": sum(int(item["frames"]) for item in clear_items),
    }

    identity_tp = sum(int(item["IDTP"]) for item in identity_items)
    identity_fn = sum(int(item["IDFN"]) for item in identity_items)
    identity_fp = sum(int(item["IDFP"]) for item in identity_items)
    identity = {
        "IDF1": float(identity_tp / max(1.0, identity_tp + 0.5 * identity_fp + 0.5 * identity_fn)),
        "IDP": float(identity_tp / max(1.0, identity_tp + identity_fp)),
        "IDR": float(identity_tp / max(1.0, identity_tp + identity_fn)),
        "IDTP": identity_tp,
        "IDFN": identity_fn,
        "IDFP": identity_fp,
    }

    hota_tp = np.sum([np.asarray(item["HOTA_TP"], dtype=np.float64) for item in hota_items], axis=0)
    hota_fn = np.sum([np.asarray(item["HOTA_FN"], dtype=np.float64) for item in hota_items], axis=0)
    hota_fp = np.sum([np.asarray(item["HOTA_FP"], dtype=np.float64) for item in hota_items], axis=0)
    association_numerator = np.sum(
        [np.asarray(item["AssA_by_threshold"]) * np.asarray(item["HOTA_TP"]) for item in hota_items], axis=0
    )
    location_numerator = np.sum(
        [np.asarray(item["LocA_by_threshold"]) * np.asarray(item["HOTA_TP"]) for item in hota_items], axis=0
    )
    association = association_numerator / np.maximum(1.0, hota_tp)
    localization = np.maximum(1e-10, location_numerator) / np.maximum(1e-10, hota_tp)
    detection = hota_tp / np.maximum(1.0, hota_tp + hota_fn + hota_fp)
    hota_values = np.sqrt(detection * association)
    thresholds = hota_items[0]["thresholds"]
    hota = {
        "HOTA": float(np.mean(hota_values)),
        "DetA": float(np.mean(detection)),
        "AssA": float(np.mean(association)),
        "LocA": float(np.mean(localization)),
        "thresholds": list(thresholds),
        "HOTA_by_threshold": [float(value) for value in hota_values],
        "DetA_by_threshold": [float(value) for value in detection],
        "AssA_by_threshold": [float(value) for value in association],
        "LocA_by_threshold": [float(value) for value in localization],
        "HOTA_TP": [int(value) for value in hota_tp],
        "HOTA_FN": [int(value) for value in hota_fn],
        "HOTA_FP": [int(value) for value in hota_fp],
    }
    return {
        "MOTA": clear["MOTA"],
        "IDF1": identity["IDF1"],
        "HOTA": hota["HOTA"],
        "clear_details": clear,
        "identity_details": identity,
        "hota_details": hota,
    }


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    return float(np.mean([np.max(precision[recall >= point], initial=0.0) for point in recall_points]))


def detection_average_precision(
    frames: Sequence[TrackingFrame],
    *,
    thresholds: Iterable[float] = np.arange(0.5, 0.96, 0.05),
) -> dict[str, object]:
    """One-class, COCO-style 101-point AP over per-frame detections."""

    frame_list = list(frames)
    threshold_values = np.asarray(list(thresholds), dtype=np.float64)
    if threshold_values.ndim != 1 or not len(threshold_values) or not np.all(np.isfinite(threshold_values)) or np.any((threshold_values <= 0) | (threshold_values > 1)):
        raise ValueError("AP thresholds must be finite nonempty values in (0, 1]")
    gt_total = sum(len(frame.gt_boxes) for frame in frame_list)
    predictions = sorted(
        (
            (float(score), frame_index, detection_index)
            for frame_index, frame in enumerate(frame_list)
            for detection_index, score in enumerate(frame.pred_scores)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    ap_values: list[float] = []
    for threshold in threshold_values:
        matched = [np.zeros(len(frame.gt_boxes), dtype=bool) for frame in frame_list]
        true_positive = np.zeros(len(predictions), dtype=np.float64)
        for index, (_, frame_index, detection_index) in enumerate(predictions):
            frame = frame_list[frame_index]
            if not len(frame.gt_boxes):
                continue
            iou = box_iou_xyxy(frame.gt_boxes, frame.pred_boxes[detection_index : detection_index + 1])[:, 0]
            iou[matched[frame_index]] = -1.0
            best = int(np.argmax(iou))
            if iou[best] >= threshold - np.finfo(float).eps:
                matched[frame_index][best] = True
                true_positive[index] = 1.0
        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(1.0 - true_positive)
        recall = cumulative_tp / max(1.0, gt_total)
        precision = cumulative_tp / np.maximum(1.0, cumulative_tp + cumulative_fp)
        ap_values.append(_interpolated_ap(recall, precision) if gt_total else 0.0)
    return {
        "AP50": float(ap_values[0]) if ap_values else 0.0,
        "AP50_95": float(np.mean(ap_values)) if ap_values else 0.0,
        "thresholds": [float(value) for value in threshold_values],
        "AP_by_threshold": ap_values,
        "ground_truth_detections": int(gt_total),
        "predicted_detections": int(len(predictions)),
    }
