from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from .dual_stream import DualStreamOBBModel
from .routing import RouteOutput, UtilityRouter


def _load_compatible_utilityroute_state(model: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    """Load matching tensors and preserve global utility weights when spatial inputs are added."""
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            continue
        if source_value.shape == target_value.shape:
            compatible[key] = source_value
            continue
        if (
            key == "router.utility.0.weight"
            and source_value.ndim == target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            merged = torch.zeros_like(target_value)
            columns = min(source_value.shape[1], target_value.shape[1])
            merged[:, :columns].copy_(source_value[:, :columns])
            compatible[key] = merged
    model.load_state_dict(compatible, strict=False)


class UtilityRouteOBBModel(DualStreamOBBModel):
    """Dual-stream OBB detector with risk-conditioned positive-utility Top-k routing."""

    def __init__(
        self,
        *args,
        beta: float = 0.015,
        top_k: int = 2,
        lambda_utility: float = 1.0,
        lambda_rank: float = 0.1,
        lambda_balance: float = 0.01,
        allow_identity: bool = True,
        use_spatial_risk: bool = False,
        spatial_grid_size: int = 4,
        **kwargs,
    ):
        weights = kwargs.get("weights", args[2] if len(args) > 2 else None)
        source_model = (weights.get("ema") or weights.get("model")) if isinstance(weights, dict) else weights
        super().__init__(*args, **kwargs)
        channels = int(getattr(self, "p3_channels", self.model[2].conv.out_channels))
        self.router = UtilityRouter(
            channels=channels,
            beta=beta,
            top_k=top_k,
            allow_identity=allow_identity,
            use_spatial_risk=use_spatial_risk,
            spatial_grid_size=spatial_grid_size,
        )
        if isinstance(source_model, UtilityRouteOBBModel):
            if getattr(source_model.router, "expert_input", None) != "fused_p3":
                warnings.warn(
                    "Loading legacy branch-input expert weights as initialization only. "
                    "Retrain the fused-P3 experts and regenerate utility targets before evaluation.",
                    UserWarning,
                    stacklevel=2,
                )
            _load_compatible_utilityroute_state(self, source_model.state_dict())
            alignment = self.router.alignment_expert
            offset_head = alignment.offset[-2]
            if (torch.count_nonzero(alignment.project.weight) == 0
                    and torch.count_nonzero(offset_head.weight) == 0
                    and torch.count_nonzero(offset_head.bias) == 0):
                # Earlier snapshots zeroed both sides of this product and could
                # never learn alignment. Repair initialization before retraining.
                torch.nn.init.dirac_(alignment.project.weight)
                warnings.warn("Reset a dead zero-initialized alignment projection to identity.", UserWarning, stacklevel=2)
        self.forced_expert: int | None = None
        self.forced_expert_mask: torch.Tensor | None = None
        self.force_all_experts = False
        self.last_route: RouteOutput | None = None
        self.lambda_utility = lambda_utility
        self.lambda_rank = lambda_rank
        self.lambda_balance = lambda_balance

    def __getstate__(self):
        state = super().__getstate__()
        # Forward diagnostics are frame-specific tensors, not checkpoint state.
        state["last_route"] = None
        state.pop("last_router_losses", None)
        return state

    def _predict_once(self, x: torch.Tensor, profile: bool = False, visualize: bool = False, embed=None):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"UtilityRouteOBBModel expects BCHW with 4 channels, got {tuple(x.shape)}")
        p3_rgb, p4_rgb = self._branch_forward(self.model[0], x[:, :3])
        p3_t, p4_t = self._branch_forward(self.model[1], x[:, 3:4])
        base_p3 = self.model[2](torch.cat([p3_rgb, p3_t], dim=1))
        route = self.router(
            p3_rgb,
            p3_t,
            base_p3,
            forced_expert=self.forced_expert,
            forced_mask=getattr(self, "forced_expert_mask", None),
            force_all=self.force_all_experts,
        )
        self.last_route = route
        p3 = route.fused
        p4 = self.model[3](torch.cat([p4_rgb, p4_t], dim=1))
        return self._tail_forward(p3, p4)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        preds = self.predict(batch["img"]) if preds is None else preds
        detection_loss, loss_items = self.criterion(preds, batch)
        if "utility_target" not in batch:
            return detection_loss, loss_items
        if self.last_route is None:
            raise RuntimeError("Utility supervision requires predictions from this model's routed forward")
        target = batch["utility_target"]
        if isinstance(target, (tuple, list)):
            target = torch.stack(target)
        target = target.to(self.last_route.predicted_utility.device, dtype=self.last_route.predicted_utility.dtype)
        predicted = self.last_route.predicted_utility
        if target.shape != predicted.shape or not torch.isfinite(target).all():
            raise ValueError(f"utility_target must be finite with shape {tuple(predicted.shape)}")
        utility_loss = F.smooth_l1_loss(predicted, target)
        target_diff = target[:, :, None] - target[:, None, :]
        pred_diff = predicted[:, :, None] - predicted[:, None, :]
        valid = target_diff.abs() > 0.01
        if valid.any():
            rank_loss = F.relu(0.05 - target_diff.sign() * pred_diff)[valid].mean()
        else:
            rank_loss = predicted.new_zeros(())
        usage = torch.softmax(self.last_route.net_score, dim=1).mean(dim=0)
        balance_loss = ((usage - 1.0 / usage.numel()) ** 2).mean()
        router_loss = (
            self.lambda_utility * utility_loss
            + self.lambda_rank * rank_loss
            + self.lambda_balance * balance_loss
        )
        self.last_router_losses = {
            "utility": utility_loss.detach(),
            "rank": rank_loss.detach(),
            "balance": balance_loss.detach(),
        }
        # Ultralytics 8.3.187 returns a three-component detection vector. Adding
        # a scalar to that vector would count the router objective three times
        # when the trainer sums it. Match its batch scaling exactly once.
        return detection_loss.sum() + router_loss * batch["img"].shape[0], loss_items
