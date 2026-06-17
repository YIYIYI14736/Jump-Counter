"""Configuration for the JumpRope MLP classifier.

IMPORTANT: NORMALIZE_STATS must stay in sync with the normalization in
jumprope_classifier.cpp (JumpRopeClassifier::normalize_input).
If you collect enough data and recompute these statistics, update BOTH
this file and the C++ source.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Feature names matching jumprope_feature.h ordering
FEATURE_NAMES: List[str] = [
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
]

FEATURE_DIM: int = len(FEATURE_NAMES)

# Normalization: (mean, std) per feature.
# Used by both Python training and C++ inference.
NORMALIZE_STATS: List[Tuple[float, float]] = [
    (22.0, 10.0),    # duration_frames
    (0.08, 0.04),    # amplitude_ratio
    (0.40, 0.12),    # rise_time_ratio
    (0.60, 0.12),    # fall_time_ratio
    (0.80, 0.40),    # rise_fall_symmetry
    (0.10, 0.06),    # knee_flexion_ratio
    (0.05, 0.04),    # ankle_elevation_ratio
    (0.50, 0.20),    # avg_confidence
    (0.03, 0.02),    # left_right_symmetry
    (0.04, 0.02),    # peak_velocity_ratio
    (30.0, 20.0),    # amplitude_pixels
    (350.0, 150.0),  # body_height_pixels
]


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = FEATURE_DIM
    hidden1: int = 16
    hidden2: int = 8
    output_dim: int = 1
    total_params: int = 353  # 12*16+16 + 16*8+8 + 8*1+1


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("training/jumprope_classifier/data")
    output_dir: Path = Path("training/jumprope_classifier/exports")
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    patience: int = 15
