"""Export the trained MLP to ONNX (and optionally NCNN via pnnx).

This script is useful when you want to integrate the classifier into
an NCNN pipeline via the standard model-file path rather than loading
raw binary weights.

Usage:
    cd JumpRope-Model
    python -m training.jumprope_classifier.export_ncnn \
        --checkpoint training/jumprope_classifier/exports/best_model.pt \
        --output-dir training/jumprope_classifier/exports
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import ModelConfig
from .train_classifier import JumpRopeMLP, export_from_pytorch


def export_onnx(checkpoint: Path, output_dir: Path, imgsz: int = 1) -> Path:
    """Export the MLP to ONNX format."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for ONNX export.")

    model = JumpRopeMLP()
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "jumprope_mlp.onnx"

    dummy = torch.randn(1, ModelConfig.input_dim)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=12,
    )
    print(f"Exported ONNX to {onnx_path}")
    return onnx_path


def print_pnnx_command(onnx_path: Path) -> None:
    """Print the pnnx command for converting ONNX → NCNN."""
    print(f"\nTo convert to NCNN with pnnx, run:")
    print(f"  pnnx {onnx_path.as_posix()} inputshape=[1,{ModelConfig.input_dim}]")
    print(f"\nOr use the raw-binary loader in C++:")
    print(f"  JumpRopeClassifier::load(\"jumprope_mlp.bin\")")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export JumpRope MLP to ONNX/NCNN.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("training/jumprope_classifier/exports"))
    parser.add_argument("--raw-only", action="store_true", help="Only export raw binary (skip ONNX)")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    # Always export raw binary for C++ direct loading
    export_from_pytorch(args.checkpoint, args.output_dir)

    if not args.raw_only:
        onnx_path = export_onnx(args.checkpoint, args.output_dir)
        print_pnnx_command(onnx_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
