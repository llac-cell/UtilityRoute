from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules import Concat, Conv
from ultralytics.nn.tasks import BaseModel, OBBModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8OBBLoss


def _thermalize_first_conv(branch: nn.ModuleList) -> None:
    block = branch[0]
    old = block.conv
    replacement = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    block.conv = replacement


def _average_init(module: Conv, channels: int) -> None:
    with torch.no_grad():
        module.conv.weight.zero_()
        for channel in range(channels):
            module.conv.weight[channel, channel, 0, 0] = 0.5
            module.conv.weight[channel, channel + channels, 0, 0] = 0.5
        if hasattr(module, "bn"):
            module.bn.weight.fill_(1.0)
            module.bn.bias.zero_()


def _fusion_indices(base: OBBModel) -> tuple[int, int]:
    """Infer the P3/P4 backbone skip indices from the first two neck concatenations."""
    skip_indices: list[int] = []
    for layer in base.model:
        if not isinstance(layer, Concat) or not isinstance(layer.f, list):
            continue
        positive = [int(index) for index in layer.f if int(index) >= 0]
        if positive:
            skip_indices.append(positive[-1])
        if len(skip_indices) == 2:
            break
    if len(skip_indices) != 2:
        raise ValueError("Unable to infer P3/P4 skips from OBB neck concatenations")
    p4_index, p3_index = skip_indices
    if not (0 <= p3_index < p4_index < len(base.model) - 1):
        raise ValueError(f"Invalid inferred fusion indices: P3={p3_index}, P4={p4_index}")
    return p3_index, p4_index


def _feature_channels(base: OBBModel, p3_index: int, p4_index: int) -> tuple[int, int]:
    """Measure backbone feature widths without assuming a YOLO generation or scale."""
    training = base.training
    base.eval()
    parameter = next(base.parameters())
    x = torch.zeros(1, 3, 64, 64, device=parameter.device, dtype=parameter.dtype)
    channels: dict[int, int] = {}
    with torch.inference_mode():
        for index, layer in enumerate(base.model[: p4_index + 1]):
            if getattr(layer, "f", -1) != -1:
                raise ValueError(f"Backbone layer {index} has unsupported non-sequential input {layer.f}")
            x = layer(x)
            if index in {p3_index, p4_index}:
                channels[index] = int(x.shape[1])
    base.train(training)
    if p3_index not in channels or p4_index not in channels:
        raise ValueError("Unable to measure inferred P3/P4 channels")
    return channels[p3_index], channels[p4_index]


def _architecture_yaml_file(cfg: str | dict, base: OBBModel) -> str:
    """Return an honest Ultralytics model label for saved/checkpoint summaries."""
    if isinstance(cfg, (str, Path)):
        return Path(cfg).name
    scale = str(base.yaml.get("scale", "s"))
    family = "yolo11" if any(type(layer).__name__ == "C2PSA" for layer in base.model) else "yolov8"
    return f"{family}{scale}-obb.yaml"


class DualStreamOBBModel(BaseModel):
    """OBB detector with independent RGB/T branches through P4 and P3/P4 fusion."""

    def __init__(self, cfg: str | dict, nc: int, weights: Any = None, verbose: bool = True):
        super().__init__()
        source_model = (weights.get("ema") or weights.get("model")) if isinstance(weights, dict) else weights
        custom_state = source_model.state_dict() if isinstance(source_model, DualStreamOBBModel) else None
        base = OBBModel(cfg, ch=3, nc=nc, verbose=verbose)
        if weights is not None and custom_state is None:
            base.load(weights, verbose=verbose)

        p3_index, p4_index = _fusion_indices(base)
        p3_channels, p4_channels = _feature_channels(base, p3_index, p4_index)
        rgb_branch = nn.ModuleList(copy.deepcopy(list(base.model[: p4_index + 1])))
        thermal_branch = nn.ModuleList(copy.deepcopy(list(base.model[: p4_index + 1])))
        _thermalize_first_conv(thermal_branch)
        fuse_p3 = Conv(p3_channels * 2, p3_channels, 1, 1)
        fuse_p4 = Conv(p4_channels * 2, p4_channels, 1, 1)
        _average_init(fuse_p3, p3_channels)
        _average_init(fuse_p4, p4_channels)
        tail = [copy.deepcopy(layer) for layer in base.model[p4_index + 1 :]]
        self.model = nn.ModuleList([rgb_branch, thermal_branch, fuse_p3, fuse_p4, *tail])

        self.fusion_p3_index = p3_index
        self.fusion_p4_index = p4_index
        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.save = list(base.save)
        self.stride = base.stride.clone()
        self.names = dict(base.names)
        self.nc = nc
        self.args = copy.deepcopy(getattr(source_model, "args", get_cfg()))
        self.yaml = copy.deepcopy(base.yaml)
        self.yaml["channels"] = 4
        self.yaml_file = _architecture_yaml_file(cfg, base)
        self.architecture = Path(self.yaml_file).stem
        self.task = "obb"
        self.inplace = getattr(base, "inplace", True)
        self.criterion = None
        if custom_state is not None:
            self.load_state_dict(custom_state, strict=False)

    def _apply(self, fn):
        super()._apply(fn)
        # BaseModel moves the head's unregistered stride/anchor tensors, but a
        # separately cloned self.stride would otherwise remain on the old device.
        self.stride = self.model[-1].stride
        self.criterion = None
        if hasattr(self, "last_route"):
            self.last_route = None
        return self

    @property
    def head(self):
        return self.model[-1]

    def _branch_forward(self, branch: nn.ModuleList, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p3_index = int(getattr(self, "fusion_p3_index", 4))
        for index, layer in enumerate(branch):
            x = layer(x)
            if index == p3_index:
                p3 = x
        return p3, x

    def _tail_forward(self, p3: torch.Tensor, p4: torch.Tensor):
        """Execute the copied Ultralytics tail from its original graph metadata."""
        p3_index = int(getattr(self, "fusion_p3_index", 4))
        p4_index = int(getattr(self, "fusion_p4_index", 6))
        outputs: dict[int, torch.Tensor] = {p3_index: p3, p4_index: p4}
        x: torch.Tensor | list[torch.Tensor] = p4
        for layer in self.model[4:]:
            source = getattr(layer, "f", -1)
            if source != -1:
                if isinstance(source, int):
                    x = outputs[source]
                else:
                    x = [x if index == -1 else outputs[int(index)] for index in source]
            x = layer(x)
            outputs[int(layer.i)] = x
        return x

    def _predict_once(self, x: torch.Tensor, profile: bool = False, visualize: bool = False, embed=None):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"DualStreamOBBModel expects BCHW with 4 channels, got {tuple(x.shape)}")
        p3_rgb, p4_rgb = self._branch_forward(self.model[0], x[:, :3])
        p3_t, p4_t = self._branch_forward(self.model[1], x[:, 3:4])
        p3 = self.model[2](torch.cat([p3_rgb, p3_t], dim=1))
        p4 = self.model[3](torch.cat([p4_rgb, p4_t], dim=1))

        return self._tail_forward(p3, p4)

    def init_criterion(self):
        return v8OBBLoss(self)
