"""Seed-level statistics; never use frames as independent repeated runs."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import t

PAPER_SEEDS = (42, 123, 3407, 2026, 9001)


def seed_statistics(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("At least two finite seed-level values are required")
    mean = float(data.mean())
    sd = float(data.std(ddof=1))
    half_width = float(t.ppf(0.975, len(data) - 1) * sd / math.sqrt(len(data)))
    return {"n": len(data), "mean": mean, "sample_sd": sd,
            "ci95": [mean - half_width, mean + half_width]}
