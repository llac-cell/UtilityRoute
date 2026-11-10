from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class ResidualExpert(nn.Module):
    def __init__(self, channels: int, depthwise: bool = False):
        super().__init__()
        groups = channels if depthwise else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlignmentExpert(nn.Module):
    def __init__(self, channels: int, max_offset: float = 4.0):
        super().__init__()
        self.max_offset = max_offset
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.offset[-2].weight)
        nn.init.zeros_(self.offset[-2].bias)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        # Zero displacement already yields a zero residual. A zero projection
        # as well would block gradients to both heads permanently.
        nn.init.dirac_(self.project.weight)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = thermal.shape
        offsets = self.offset(torch.cat([rgb, thermal], dim=1))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=thermal.device, dtype=thermal.dtype),
            torch.linspace(-1, 1, width, device=thermal.device, dtype=thermal.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1).clone()
        grid[..., 0] += offsets[:, 0] * (2 * self.max_offset / max(width - 1, 1))
        grid[..., 1] += offsets[:, 1] * (2 * self.max_offset / max(height - 1, 1))
        warped = F.grid_sample(thermal, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return self.project(warped - thermal)


class RiskEstimator(nn.Module):
    def __init__(self, channels: int, global_dim: int = 128):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.spatial(fused)), self.global_head(fused)


@dataclass
class RouteOutput:
    fused: torch.Tensor
    predicted_utility: torch.Tensor
    net_score: torch.Tensor
    active_mask: torch.Tensor
    spatial_risk: torch.Tensor


class UtilityRouter(nn.Module):
    expert_names = ("rgb_illumination", "thermal_contrast", "weather_noise", "alignment")

    def __init__(
        self,
        channels: int = 128,
        beta: float = 0.015,
        top_k: int = 2,
        allow_identity: bool = True,
        use_spatial_risk: bool = False,
        spatial_grid_size: int = 4,
    ):
        super().__init__()
        if spatial_grid_size <= 0:
            raise ValueError("spatial_grid_size must be positive")
        if not isinstance(top_k, int) or not 1 <= top_k <= len(self.expert_names):
            raise ValueError("top_k must be an integer from 1 to 4")
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        # Changing branch inputs to fused P3 changes the meaning of learned weights.
        # Keep a contract on the serialized module so old full-object checkpoints
        # cannot silently be evaluated under a different expert implementation.
        self.implementation_contract = "manuscript_20261108"
        self.expert_input = "fused_p3"
        self.beta = beta
        self.top_k = top_k
        self.allow_identity = allow_identity
        self.use_spatial_risk = use_spatial_risk
        self.spatial_grid_size = spatial_grid_size
        self.risk = RiskEstimator(channels)
        spatial_features = spatial_grid_size**2 if use_spatial_risk else 0
        self.utility = nn.Sequential(
            nn.Linear(128 + channels + spatial_features, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, len(self.expert_names)),
        )
        self.rgb_expert = ResidualExpert(channels)
        self.thermal_expert = ResidualExpert(channels)
        self.weather_expert = ResidualExpert(channels, depthwise=True)
        self.alignment_expert = AlignmentExpert(channels)
        self.register_buffer("normalized_cost", torch.tensor([0.55, 0.55, 0.75, 1.0]))

    @staticmethod
    def _pool_spatial_grid(spatial: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Use deterministic block means for the divisible feature maps used by this experiment."""
        batch, channels, height, width = spatial.shape
        if height % grid_size == 0 and width % grid_size == 0:
            return spatial.reshape(
                batch,
                channels,
                grid_size,
                height // grid_size,
                grid_size,
                width // grid_size,
            ).mean(dim=(3, 5))
        return F.adaptive_avg_pool2d(spatial, (grid_size, grid_size))

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        base_fused: torch.Tensor,
        forced_expert: int | None = None,
        force_all: bool = False,
        forced_mask: torch.Tensor | None = None,
    ) -> RouteOutput:
        spatial, global_risk = self.risk(base_fused)
        pooled = F.adaptive_avg_pool2d(base_fused, 1).flatten(1)
        utility_features = [global_risk, pooled]
        if getattr(self, "use_spatial_risk", False):
            grid_size = int(getattr(self, "spatial_grid_size", 4))
            utility_features.append(self._pool_spatial_grid(spatial, grid_size).flatten(1))
        predicted = self.utility(torch.cat(utility_features, dim=1))
        score = predicted - self.beta * self.normalized_cost
        controls = int(forced_expert is not None) + int(forced_mask is not None) + int(force_all)
        if controls > 1:
            raise ValueError("Use only one of forced_expert, forced_mask, and force_all")
        if getattr(self, "expert_input", None) != "fused_p3":
            raise RuntimeError(
                "Legacy router checkpoint uses branch-input experts. Reconstruct the model "
                "with UtilityRouteOBBModel, retrain the fused-P3 experts, and regenerate "
                "counterfactual targets before evaluating the latest manuscript contract."
            )
        if forced_expert is not None:
            if forced_expert not in (-1, 0, 1, 2, 3):
                raise ValueError("forced_expert must be -1 (Identity) or one of 0,1,2,3")
            mask = torch.zeros_like(score)
            if forced_expert >= 0:
                mask[:, forced_expert] = 1.0
        elif forced_mask is not None:
            mask = torch.as_tensor(forced_mask, device=score.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2 or mask.shape[1] != score.shape[1]:
                raise ValueError(
                    f"forced_mask must have shape (4,), (1, 4), or (batch, 4); got {tuple(mask.shape)}"
                )
            if mask.shape[0] == 1:
                mask = mask.expand(score.shape[0], -1)
            elif mask.shape[0] != score.shape[0]:
                raise ValueError(
                    f"forced_mask batch dimension must be 1 or {score.shape[0]}; got {mask.shape[0]}"
                )
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("forced_mask values must be binary (0 or 1)")
            mask = mask.to(dtype=score.dtype)
        elif force_all:
            mask = torch.ones_like(score)
        else:
            values, indices = torch.topk(score, k=min(self.top_k, score.shape[1]), dim=1)
            selected = (values > 0).to(score.dtype) if getattr(self, "allow_identity", True) else torch.ones_like(values)
            mask = torch.zeros_like(score).scatter(1, indices, selected)
        # Dispatch only the selected samples to each expert. Computing every
        # residual and masking afterward does not save expert computation.
        correction = torch.zeros_like(base_fused)
        experts = (self.rgb_expert, self.thermal_expert, self.weather_expert, self.alignment_expert)
        for expert_index, expert in enumerate(experts):
            indices = torch.nonzero(mask[:, expert_index] > 0, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            if expert_index == 3:
                residual = expert(rgb.index_select(0, indices), thermal.index_select(0, indices))
            else:
                residual = expert(base_fused.index_select(0, indices))
            correction = correction.index_add(0, indices, residual)
        count = mask.sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
        return RouteOutput(base_fused + correction / count, predicted, score, mask, spatial)
