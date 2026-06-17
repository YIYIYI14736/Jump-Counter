"""Augment sparse pseudo-labeled jump-rope feature CSVs into a balanced dataset.

Usage:
    cd JumpRope-Model
    python -m training.jumprope_classifier.augment_data ^
        --inputs ../jumprope_20260614_231202.csv ../jumprope_20260614_232330.csv ../jumprope_20260614_232447.csv ^
        --output training/jumprope_classifier/data/jumprope_augmented.csv ^
        --target-per-class 500

Augmentation primitives (all keep features physically plausible):
  1. Gaussian jitter scaled per-feature std.
  2. Rise/fall swap: rise_time_ratio <-> fall_time_ratio, rise_fall_symmetry -> 1 / x.
  3. Integer/positive clamping for duration_frames.
  4. Clip all features to safe ranges (no negatives on ratio features).

The output CSV keeps the exact header/layout consumed by train_classifier.load_csv_files.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import FEATURE_DIM, FEATURE_NAMES


# Indices that should remain non-negative after augmentation.
RATIO_FEATURES = {
    "duration_frames",
    "amplitude_ratio",
    "rise_time_ratio",
    "fall_time_ratio",
    "rise_fall_symmetry",
    "knee_flexion_ratio",
    "ankle_elevation_ratio",
    "avg_confidence",
    "left_right_symmetry",
    "peak_velocity_ratio",
    "amplitude_pixels",
    "body_height_pixels",
}

RISE_IDX = FEATURE_NAMES.index("rise_time_ratio")
FALL_IDX = FEATURE_NAMES.index("fall_time_ratio")
SYM_IDX = FEATURE_NAMES.index("rise_fall_symmetry")
DUR_IDX = FEATURE_NAMES.index("duration_frames")


def load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows_X: List[List[float]] = []
    rows_y: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < FEATURE_DIM + 1:
                continue
            label = int(row[0])
            feats = [float(v) for v in row[1 : FEATURE_DIM + 1]]
            if len(feats) != FEATURE_DIM:
                continue
            rows_X.append(feats)
            rows_y.append(label)
    return np.asarray(rows_X, dtype=np.float32), np.asarray(rows_y, dtype=np.int64)


def clip_safe(x: np.ndarray) -> np.ndarray:
    """Clamp each feature to a small non-negative floor so ratios stay physical."""
    out = x.copy()
    for j, name in enumerate(FEATURE_NAMES):
        if name == "duration_frames":
            out[:, j] = np.clip(out[:, j], 1.0, 1e4)
        elif name in ("avg_confidence",):
            out[:, j] = np.clip(out[:, j], 0.0, 1.0)
        elif name in RATIO_FEATURES:
            # Ratios / pixels: never negative.
            out[:, j] = np.clip(out[:, j], 0.0, None)
    return out


def jitter(x: np.ndarray, stds: np.ndarray, rng: np.random.RandomState, scale: float = 0.08) -> np.ndarray:
    noise = rng.randn(*x.shape).astype(np.float32) * (stds * scale)
    return x + noise


def rise_fall_swap(x: np.ndarray) -> np.ndarray:
    """Swap rise/fall ratios and invert symmetry. Mirrors a valid motion."""
    out = x.copy()
    out[:, RISE_IDX], out[:, FALL_IDX] = x[:, FALL_IDX].copy(), x[:, RISE_IDX].copy()
    sym = out[:, SYM_IDX]
    sym_safe = np.where(sym > 1e-3, sym, 1e-3)
    out[:, SYM_IDX] = 1.0 / sym_safe
    return out


def augment_class(
    X: np.ndarray,
    target: int,
    stds: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Build a (target, FEATURE_DIM) array for one class via jitter + swaps."""
    n = len(X)
    if n == 0:
        raise ValueError("Cannot augment an empty class.")

    out: List[np.ndarray] = [X.copy()]
    needed = max(target - n, 0)

    while needed > 0:
        batch = min(needed, n)
        idx = rng.randint(0, n, size=batch)
        base = X[idx]

        # Alternate augmentation operators for diversity.
        op = rng.randint(0, 3)
        if op == 0:
            aug = jitter(base, stds, rng, scale=0.06)
        elif op == 1:
            aug = rise_fall_swap(jitter(base, stds, rng, scale=0.04))
        else:
            aug = jitter(rise_fall_swap(base), stds, rng, scale=0.04)

        aug = clip_safe(aug)
        # duration_frames is an integer count.
        aug[:, DUR_IDX] = np.maximum(np.round(aug[:, DUR_IDX]), 1.0)

        out.append(aug)
        needed -= batch

    return np.concatenate(out, axis=0)[:target]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment JumpRope feature CSVs into a balanced dataset.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Input CSV files.")
    p.add_argument("--output", type=Path, required=True, help="Output augmented CSV.")
    p.add_argument("--target-per-class", type=int, default=500)
    p.add_argument("--jitter-clip-quantile", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        Xc, yc = load_csv(path)
        print(f"  {path.name}: {len(yc)} rows ({int((yc == 1).sum())} pos, {int((yc == 0).sum())} neg)")
        X_parts.append(Xc)
        y_parts.append(yc)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f"Total raw: {len(y)} ({int((y == 1).sum())} pos, {int((y == 0).sum())} neg)")

    # Per-feature std used for jitter; fall back to a small constant if a class is degenerate.
    overall_std = X.std(axis=0)
    overall_std = np.where(overall_std > 1e-6, overall_std, 1e-2)

    rng = np.random.RandomState(args.seed)

    X_pos = X[y == 1]
    X_neg = X[y == 0]
    X_pos_aug = augment_class(X_pos, args.target_per_class, overall_std, rng)
    X_neg_aug = augment_class(X_neg, args.target_per_class, overall_std, rng)
    print(f"Augmented: pos={len(X_pos_aug)}, neg={len(X_neg_aug)}")

    X_out = np.concatenate([X_pos_aug, X_neg_aug], axis=0)
    y_out = np.concatenate(
        [np.ones(len(X_pos_aug), dtype=np.int64), np.zeros(len(X_neg_aug), dtype=np.int64)]
    )

    # Shuffle so positives/negatives interleave.
    perm = rng.permutation(len(y_out))
    X_out = X_out[perm]
    y_out = y_out[perm]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + FEATURE_NAMES)
        for label, feats in zip(y_out, X_out):
            writer.writerow([int(label)] + [f"{v:.6f}" for v in feats])

    print(f"Wrote {len(y_out)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
