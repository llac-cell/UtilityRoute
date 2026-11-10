from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import cv2
import numpy as np
import torch


SEVERITY_LEVELS = (1, 2, 3, 4, 5)

SINGLE_CONDITIONS = (
    "low_light",
    "overexposure",
    "fog",
    "motion_blur",
    "rgb_noise",
    "thermal_low_contrast",
    "rgbt_translation",
    "occlusion",
)

COMPOUND_CONDITIONS: dict[str, tuple[str, str]] = {
    "low_light_motion_blur": ("low_light", "motion_blur"),
    "fog_thermal_low_contrast": ("fog", "thermal_low_contrast"),
    "rgb_noise_rgbt_translation": ("rgb_noise", "rgbt_translation"),
}

CONDITIONS = (*SINGLE_CONDITIONS, *COMPOUND_CONDITIONS)

SEVERITY_PARAMETERS: dict[str, Sequence[float | int]] = {
    "low_light": (1.5, 2.0, 2.5, 3.0, 3.5),
    "overexposure": (0.80, 0.65, 0.50, 0.40, 0.30),
    "fog": (0.85, 0.70, 0.55, 0.40, 0.25),
    "motion_blur": (3, 5, 9, 13, 17),
    "rgb_noise": (5, 10, 20, 30, 40),
    "thermal_low_contrast": (0.80, 0.65, 0.50, 0.35, 0.20),
    "rgbt_translation": (2, 4, 8, 12, 16),
    "occlusion": (0.05, 0.10, 0.20, 0.30, 0.40),
}


def deterministic_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _to_float_numpy(image: torch.Tensor) -> tuple[np.ndarray, float]:
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 4:
        shape = getattr(image, "shape", None)
        raise ValueError(f"Expected a 4-channel CHW tensor, got {shape}")
    array = image.detach().cpu().numpy().astype(np.float32, copy=True)
    scale = 1.0 if image.is_floating_point() else 255.0
    if scale != 1.0:
        array /= scale
    return np.clip(array, 0.0, 1.0), scale


def _from_float_numpy(array: np.ndarray, reference: torch.Tensor, scale: float) -> torch.Tensor:
    array = np.clip(array, 0.0, 1.0)
    if reference.is_floating_point():
        return torch.from_numpy(array).to(device=reference.device, dtype=reference.dtype)
    rounded = np.rint(array * scale)
    return torch.from_numpy(rounded).to(device=reference.device, dtype=reference.dtype)


def _motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = center
    angle = math.radians(angle_degrees)
    dx = math.cos(angle) * radius
    dy = math.sin(angle) * radius
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, color=1.0, thickness=1)
    total = float(kernel.sum())
    if total == 0:
        kernel[size // 2, size // 2] = 1.0
        total = 1.0
    return kernel / total


def _apply_family(
    array: np.ndarray,
    family: str,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = array.copy()
    value = SEVERITY_PARAMETERS[family][severity - 1]

    if family == "low_light":
        output[:3] = np.power(output[:3], float(value))
    elif family == "overexposure":
        output[:3] = np.power(output[:3], float(value))
    elif family == "fog":
        transmission = float(value)
        output[:3] = output[:3] * transmission + (1.0 - transmission)
    elif family == "motion_blur":
        kernel = _motion_kernel(int(value), float(rng.uniform(0.0, 180.0)))
        for channel in range(3):
            output[channel] = cv2.filter2D(output[channel], -1, kernel, borderType=cv2.BORDER_REPLICATE)
    elif family == "rgb_noise":
        sigma = float(value) / 255.0
        noise = rng.normal(0.0, sigma, size=output[:3].shape).astype(np.float32)
        output[:3] += noise
    elif family == "thermal_low_contrast":
        alpha = float(value)
        mean = float(output[3].mean())
        output[3] = mean + alpha * (output[3] - mean)
    elif family == "rgbt_translation":
        magnitude = int(value)
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        direction_x, direction_y = directions[int(rng.integers(0, len(directions)))]
        transform = np.float32([[1, 0, direction_x * magnitude], [0, 1, direction_y * magnitude]])
        height, width = output.shape[1:]
        output[3] = cv2.warpAffine(
            output[3],
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    elif family == "occlusion":
        ratio = float(value)
        height, width = output.shape[1:]
        aspect = float(np.exp(rng.uniform(math.log(0.5), math.log(2.0))))
        rectangle_h = max(1, min(height, int(round(math.sqrt(height * width * ratio / aspect)))))
        rectangle_w = max(1, min(width, int(round(math.sqrt(height * width * ratio * aspect)))))
        top = int(rng.integers(0, max(height - rectangle_h + 1, 1)))
        left = int(rng.integers(0, max(width - rectangle_w + 1, 1)))
        fill = output.reshape(4, -1).mean(axis=1)
        output[:, top : top + rectangle_h, left : left + rectangle_w] = fill[:, None, None]
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return np.clip(output, 0.0, 1.0)


def apply_condition(
    image: torch.Tensor,
    condition: str,
    severity: int,
    *,
    sample_id: str,
    base_seed: int = 42,
) -> torch.Tensor:
    """Apply one deterministic, in-memory RGB--T corruption condition."""
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {severity}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown corruption condition: {condition}")

    array, scale = _to_float_numpy(image)
    families = COMPOUND_CONDITIONS.get(condition, (condition,))
    for component_index, family in enumerate(families):
        seed = deterministic_seed(base_seed, sample_id, condition, severity, component_index, family)
        array = _apply_family(array, family, severity, np.random.default_rng(seed))
    return _from_float_numpy(array, image, scale)
