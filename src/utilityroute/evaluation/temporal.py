"""Independent tracking-time utility smoothing; never a fifth core expert.

The paper does not specify the temporal state/update equation. This explicit EMA
is a configurable implementation choice, not a recovery of the historical runs.
It uses only past/current predictions and resets at every sequence boundary.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalUtilityEMA:
    """Replace a utility head output with an exponential moving average.

    ``alpha`` weights the current frame, and the first frame is unchanged. The
    context must enclose a single ordered sequence with inference batch size one.
    """

    def __init__(self, utility_head: nn.Module, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("temporal alpha must be in (0, 1]")
        self.utility_head = utility_head
        self.alpha = float(alpha)
        self.state: torch.Tensor | None = None
        self.handle = None

    def _hook(self, _module, _inputs, output: torch.Tensor) -> torch.Tensor:
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("Temporal routing requires ordered batch-1 inference")
        if self.state is None:
            value = output
        else:
            value = self.alpha * output + (1.0 - self.alpha) * self.state.to(output)
        self.state = value.detach().clone()
        return value

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Temporal utility context is already active")
        self.state = None
        self.handle = self.utility_head.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.state = None
