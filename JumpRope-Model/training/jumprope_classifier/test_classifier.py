"""Tests for the JumpRope MLP classifier configuration and data utilities."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.jumprope_classifier.config import (
    FEATURE_DIM,
    FEATURE_NAMES,
    NORMALIZE_STATS,
    ModelConfig,
    TrainConfig,
)
from training.jumprope_classifier.train_classifier import (
    load_csv_files,
    normalize_features,
    train_val_split,
)


class ConfigTests(unittest.TestCase):

    def test_feature_dim_matches_names(self):
        self.assertEqual(FEATURE_DIM, len(FEATURE_NAMES))
        self.assertEqual(FEATURE_DIM, 12)

    def test_normalize_stats_length(self):
        self.assertEqual(len(NORMALIZE_STATS), FEATURE_DIM)
        for mean, std in NORMALIZE_STATS:
            self.assertGreater(std, 0, "std must be positive")

    def test_model_config_param_count(self):
        mc = ModelConfig()
        expected = (
            mc.input_dim * mc.hidden1 + mc.hidden1
            + mc.hidden1 * mc.hidden2 + mc.hidden2
            + mc.hidden2 * mc.output_dim + mc.output_dim
        )
        self.assertEqual(mc.total_params, expected)
        self.assertEqual(mc.total_params, 353)

    def test_train_config_defaults(self):
        tc = TrainConfig()
        self.assertGreater(tc.epochs, 0)
        self.assertGreater(tc.batch_size, 0)
        self.assertGreater(tc.lr, 0)
        self.assertGreater(tc.val_split, 0)
        self.assertLess(tc.val_split, 1)


class DataLoadingTests(unittest.TestCase):

    def _write_csv(self, path: Path, n_pos: int, n_neg: int):
        header = ["label"] + FEATURE_NAMES
        rows = []
        rng = np.random.RandomState(42)
        for _ in range(n_pos):
            row = [1] + rng.uniform(0, 1, FEATURE_DIM).tolist()
            rows.append(row)
        for _ in range(n_neg):
            row = [0] + rng.uniform(0, 1, FEATURE_DIM).tolist()
            rows.append(row)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def test_load_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._write_csv(data_dir / "session1.csv", n_pos=20, n_neg=10)
            self._write_csv(data_dir / "session2.csv", n_pos=15, n_neg=5)

            X, y = load_csv_files(data_dir)
            self.assertEqual(X.shape[0], 50)
            self.assertEqual(X.shape[1], FEATURE_DIM)
            self.assertEqual(int(y.sum()), 35)

    def test_load_csv_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_csv_files(Path("/nonexistent_dir_xyz"))

    def test_normalize_features(self):
        X = np.ones((5, FEATURE_DIM), dtype=np.float32)
        X_norm = normalize_features(X)
        self.assertEqual(X_norm.shape, X.shape)
        # After normalization, values should not be all ones
        self.assertFalse(np.allclose(X_norm, 1.0))

    def test_train_val_split(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, FEATURE_DIM).astype(np.float32)
        y = np.array([1] * 60 + [0] * 40, dtype=np.float32)

        X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)
        self.assertEqual(len(X_tr) + len(X_val), 100)
        self.assertEqual(len(y_tr) + len(y_val), 100)
        # Both splits should have both classes
        self.assertIn(1.0, y_tr)
        self.assertIn(0.0, y_tr)


if __name__ == "__main__":
    unittest.main()
