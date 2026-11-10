from __future__ import annotations

import copy
import hashlib
import torch
import yaml
from pathlib import Path
from typing import Any

from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from utilityroute.data import PairedYOLODataset, UtilityPairedYOLODataset
from utilityroute.models import UtilityRouteOBBModel

from .paired_trainer import PairedOBBTrainer


class UtilityRouteTrainer(PairedOBBTrainer):
    def __init__(
        self,
        *args,
        route_mode: str,
        utility_cache: str | Path | None = None,
        model_options: dict[str, Any] | None = None,
        freeze_frozen_module_stats: bool = True,
        target_mode: str = "offline",
        **kwargs,
    ):
        if route_mode not in {"experts", "router"}:
            raise ValueError(route_mode)
        if target_mode not in {"offline", "online"}:
            raise ValueError("target_mode must be offline or online")
        self.target_mode = target_mode
        self.utility_teacher = None
        self.route_mode = route_mode
        self.utility_cache = None if utility_cache is None else str(utility_cache)
        initial_weights = Path(str(kwargs.get("overrides", {}).get("model", "")))
        self.cache_checkpoint_sha256 = (
            hashlib.sha256(initial_weights.read_bytes()).hexdigest()
            if route_mode == "router" and target_mode == "offline" and initial_weights.is_file() else None
        )
        self.model_options = dict(model_options or {})
        # Kept as a compatibility argument; frozen running stats are mandatory.
        self.freeze_frozen_module_stats = True
        self.optimizer_steps = 0
        super().__init__(*args, **kwargs)
        if RANK in {-1, 0}:
            (self.save_dir / "routing_config.yaml").write_text(yaml.safe_dump({
                "route_mode": self.route_mode, "target_mode": self.target_mode,
                "utility_cache": self.utility_cache, "cache_checkpoint_sha256": self.cache_checkpoint_sha256,
                "model_options": self.model_options, "frozen_module_stats": True,
            }, sort_keys=False), encoding="utf-8")

    def get_model(self, cfg=None, weights=None, verbose=True):
        source = weights.get("model") if isinstance(weights, dict) else weights
        if self.route_mode == "router" and not isinstance(source, UtilityRouteOBBModel):
            raise TypeError("Router training requires a trained expert-bank checkpoint; run --mode experts first")
        model = UtilityRouteOBBModel(
            cfg,
            nc=self.data["nc"],
            weights=weights,
            verbose=verbose and RANK == -1,
            **self.model_options,
        )
        if self.route_mode == "router" and self.target_mode == "online":
            self.utility_teacher = copy.deepcopy(model).float().eval()
            self.utility_teacher.requires_grad_(False)
        model.force_all_experts = self.route_mode == "experts"
        model.forced_expert = None
        self._set_stage_gradients(model)
        return model

    def build_optimizer(self, model, *args, **kwargs):
        # Ultralytics re-enables requires_grad before optimizer construction.
        # Restore stage ownership before collecting optimizer parameters.
        self._set_stage_gradients(de_parallel(model))
        optimizer = super().build_optimizer(model, *args, **kwargs)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        return optimizer

    def _set_stage_gradients(self, model):
        model.model.requires_grad_(False)
        model.router.requires_grad_(False)
        if self.route_mode == "experts":
            for module in (model.router.rgb_expert, model.router.thermal_expert,
                           model.router.weather_expert, model.router.alignment_expert):
                module.requires_grad_(True)
        else:
            model.router.risk.requires_grad_(True)
            model.router.utility.requires_grad_(True)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.route_mode != "router" or self.target_mode != "online":
            return batch
        teacher = self.utility_teacher
        teacher.to(batch["img"].device).eval()
        teacher.args = de_parallel(self.model).args
        if teacher.criterion is None:
            teacher.criterion = teacher.init_criterion()
        targets = []
        with torch.no_grad():
            for index in range(batch["img"].shape[0]):
                select = batch["batch_idx"] == index
                sample = {"img": batch["img"][index:index + 1].float(),
                          "batch_idx": torch.zeros_like(batch["batch_idx"][select]),
                          "cls": batch["cls"][select], "bboxes": batch["bboxes"][select]}
                losses = []
                teacher.force_all_experts = False
                teacher.forced_expert_mask = None
                for expert in (-1, 0, 1, 2, 3):
                    teacher.forced_expert = expert
                    predictions = teacher.predict(sample["img"])
                    components, _ = teacher.criterion(predictions, sample)
                    losses.append(components.sum())
                identity = losses[0]
                targets.append(torch.stack([(identity - value) / (identity + 1e-6) for value in losses[1:]]))
        teacher.forced_expert = None
        batch["utility_target"] = torch.stack(targets).detach()
        if not torch.isfinite(batch["utility_target"]).all():
            raise ValueError("Online teacher produced non-finite utility targets")
        return batch

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self._set_stage_gradients(de_parallel(self.model))

    def _model_train(self):
        super()._model_train()
        model = de_parallel(self.model)
        # Frozen parameters must also have frozen BN running statistics.
        model.model.eval()
        if self.route_mode == "router":
            for module in (
                model.router.rgb_expert,
                model.router.thermal_expert,
                model.router.weather_expert,
                model.router.alignment_expert,
            ):
                module.eval()
        else:
            model.router.risk.eval()
            model.router.utility.eval()

    def optimizer_step(self):
        super().optimizer_step()
        self.optimizer_steps += 1

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        kwargs = dict(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train" and not (self.route_mode == "router" and self.target_mode == "offline"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="obb",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        if self.route_mode == "router" and self.target_mode == "offline":
            if self.utility_cache is None:
                raise ValueError("router mode requires utility_cache")
            return UtilityPairedYOLODataset(**kwargs, utility_cache=self.utility_cache, utility_split=mode,
                                           expected_checkpoint_sha256=self.cache_checkpoint_sha256)
        return PairedYOLODataset(**kwargs)
