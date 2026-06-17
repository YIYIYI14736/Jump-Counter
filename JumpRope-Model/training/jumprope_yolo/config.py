from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


JUMPROPE_CLASSES = ("person",)
DEFAULT_CLASSES = JUMPROPE_CLASSES


def parse_class_names(value: str) -> Tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate class name: {name}")
        seen.add(name)
    if not names:
        raise ValueError("At least one class name is required")
    return names


@dataclass(frozen=True)
class TrainOptions:
    data: Path = Path("datasets/jumprope_pose.yaml")
    model: str = "yolo11n-pose.pt"
    epochs: int = 60
    imgsz: int = 640
    batch: int = -1
    project: Path = Path("training/jumprope_yolo/runs")
    name: str = "jumprope-yolo11n-pose"
    device: Optional[str] = None
    workers: int = 2
    seed: int = 7
    patience: int = 20
    cache: bool = False
    amp: bool = True
    small_data: bool = True
    freeze: int = 0


def build_train_overrides(options: TrainOptions) -> dict:
    overrides = {
        "data": options.data.as_posix(),
        "model": options.model,
        "epochs": options.epochs,
        "imgsz": options.imgsz,
        "batch": options.batch,
        "project": options.project.as_posix(),
        "name": options.name,
        "workers": options.workers,
        "seed": options.seed,
        "patience": options.patience,
        "cache": options.cache,
        "amp": options.amp,
        "freeze": options.freeze,
        "cos_lr": True,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    if options.device:
        overrides["device"] = options.device
    if options.small_data:
        overrides.update(
            {
                "hsv_h": 0.015,
                "hsv_s": 0.5,
                "hsv_v": 0.35,
                "degrees": 3.0,
                "translate": 0.08,
                "scale": 0.35,
                "fliplr": 0.5,
            }
        )
    return overrides
