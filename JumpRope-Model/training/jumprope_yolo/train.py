import argparse
from pathlib import Path
from typing import Optional

from .config import TrainOptions, build_train_overrides


def parse_args(argv: Optional[list] = None) -> TrainOptions:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight YOLO pose model for JumpRope counting.")
    parser.add_argument("--data", type=Path, default=Path("datasets/jumprope_pose.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--project", type=Path, default=Path("training/jumprope_yolo/runs"))
    parser.add_argument("--name", default="jumprope-yolo11n-pose")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--full-aug", action="store_false", dest="small_data")
    parser.add_argument("--freeze", type=int, default=0)
    args = parser.parse_args(argv)
    return TrainOptions(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        small_data=args.small_data,
        freeze=args.freeze,
    )


def train(options: TrainOptions):
    from ultralytics import YOLO

    overrides = build_train_overrides(options)
    model = YOLO(options.model)
    return model.train(**overrides)


def main(argv: Optional[list] = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
