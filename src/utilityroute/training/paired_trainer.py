from __future__ import annotations

from copy import copy

from ultralytics.data.build import build_dataloader
from ultralytics.models import yolo
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

from utilityroute.data import PairedYOLODataset
from utilityroute.data.config import resolve_data_config
from utilityroute.models import DualStreamOBBModel


class LocalOBBTrainer(OBBTrainer):
    def get_dataset(self):
        data = resolve_data_config(self.args.data)
        if self.args.single_cls:
            data["names"], data["nc"] = {0: "item"}, 1
        return data

    def build_dataset(self, *args, **kwargs):
        dataset = super().build_dataset(*args, **kwargs)
        dataset.transforms.transforms = [t for t in dataset.transforms.transforms
                                         if type(t).__name__ != "Albumentations"]
        return dataset


class PairedOBBTrainer(LocalOBBTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {} if overrides is None else overrides
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        return DualStreamOBBModel(cfg, nc=self.data["nc"], weights=weights, verbose=verbose and RANK == -1)

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return PairedYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
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

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train" and not getattr(dataset, "rect", False)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
