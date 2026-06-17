import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ExportOptions:
    weights: Path
    imgsz: int = 640
    half: bool = False
    simplify: bool = True
    opset: int = 12
    device: str = "cpu"
    output_dir: Path = Path("training/jumprope_yolo/exports")
    format: str = "ncnn"
    ncnn_backend: str = "ultralytics"
    dynamic: bool = True


def build_pnnx_command(torchscript_path: Path, options: ExportOptions) -> List[str]:
    command = [
        "pnnx",
        torchscript_path.as_posix(),
        f"inputshape=[1,3,{options.imgsz},{options.imgsz}]",
    ]
    if options.dynamic:
        half_imgsz = max(32, options.imgsz // 2)
        command.append(f"inputshape2=[1,3,{half_imgsz},{half_imgsz}]")
    command.append(f"device={options.device}")
    return command


def export_with_ultralytics(options: ExportOptions):
    from ultralytics import YOLO

    model = YOLO(str(options.weights))
    return model.export(
        format=options.format,
        imgsz=options.imgsz,
        half=options.half,
        simplify=options.simplify,
        opset=options.opset,
        device=options.device,
        project=options.output_dir.as_posix(),
    )


def parse_args(argv: Optional[list] = None) -> ExportOptions:
    parser = argparse.ArgumentParser(description="Export JumpRope YOLO pose weights for Android/NCNN.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("training/jumprope_yolo/exports"))
    parser.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "torchscript"])
    parser.add_argument("--backend", default="ultralytics", choices=["ultralytics", "pnnx"], dest="ncnn_backend")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify")
    parser.add_argument("--static", action="store_false", dest="dynamic")
    args = parser.parse_args(argv)
    return ExportOptions(
        weights=args.weights,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        format=args.format,
        ncnn_backend=args.ncnn_backend,
        half=args.half,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )


def main(argv: Optional[list] = None) -> int:
    options = parse_args(argv)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.ncnn_backend == "pnnx":
        torchscript_path = options.output_dir / f"{options.weights.stem}.torchscript"
        print(" ".join(build_pnnx_command(torchscript_path, options)))
        return 0
    export_with_ultralytics(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
