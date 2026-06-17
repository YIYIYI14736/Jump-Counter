"""Train a small MLP classifier on pseudo-labeled jump-rope features.

Usage:
    cd JumpRope-Model
    python -m training.jumprope_classifier.train_classifier \
        --data-dir training/jumprope_classifier/data \
        --epochs 80

The data directory should contain one or more CSV files matching the format
produced by the Android recording feature:
    label,duration_frames,amplitude_ratio,...,body_height_pixels

Label 1 = valid jump (positive), 0 = false positive (negative).
"""

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Try importing PyTorch; fall back to pure-NumPy training if unavailable.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    FEATURE_DIM,
    FEATURE_NAMES,
    NORMALIZE_STATS,
    ModelConfig,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_files(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all CSV files from data_dir and concatenate."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Record data from the Android app first."
        )

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            for row_num, row in enumerate(reader, 2):
                if len(row) < FEATURE_DIM + 1:
                    print(f"  WARNING: {csv_file.name}:{row_num} has {len(row)} columns, expected {FEATURE_DIM + 1}")
                    continue

                label = int(row[0])
                features = np.array([float(v) for v in row[1 : FEATURE_DIM + 1]], dtype=np.float32)
                all_features.append(features)
                all_labels.append(label)

    if not all_features:
        raise ValueError("No valid samples found in CSV files.")

    X = np.stack(all_features)
    y = np.array(all_labels, dtype=np.float32)
    print(f"Loaded {len(y)} samples ({int(y.sum())} positive, {int(len(y) - y.sum())} negative) from {len(csv_files)} file(s)")
    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Apply z-score normalization matching C++ inference."""
    means = np.array([s[0] for s in NORMALIZE_STATS], dtype=np.float32)
    stds = np.array([s[1] for s in NORMALIZE_STATS], dtype=np.float32)
    return (X - means) / stds


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy / precision / recall / F1 + confusion matrix counts."""
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
    }


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random stratified split."""
    rng = random.Random(seed)
    pos_idx = [i for i in range(len(y)) if y[i] == 1]
    neg_idx = [i for i in range(len(y)) if y[i] == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * val_ratio))
    n_val_neg = max(1, int(len(neg_idx) * val_ratio))

    val_idx = pos_idx[:n_val_pos] + neg_idx[:n_val_neg]
    train_idx = pos_idx[n_val_pos:] + neg_idx[n_val_neg:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class JumpRopeMLP(nn.Module):
        def __init__(self, config: ModelConfig = ModelConfig()):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden1),
                nn.ReLU(),
                nn.Linear(config.hidden1, config.hidden2),
                nn.ReLU(),
                nn.Linear(config.hidden2, config.output_dim),
            )

        def forward(self, x):
            return self.net(x)

    def train_pytorch(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainConfig,
    ) -> Path:
        """Train with PyTorch and return path to best checkpoint."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        model = JumpRopeMLP().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, config.epochs + 1):
            # Train
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            scheduler.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += (preds == yb).sum().item()
                    total += len(yb)

            val_acc = correct / max(total, 1)
            train_loss = total_loss / max(len(train_ds), 1)
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                config.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.output_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
        return config.output_dir / "best_model.pt"


    def predict_pytorch(checkpoint_path: Path, X: np.ndarray) -> np.ndarray:
        """Load a checkpoint and return 0/1 predictions for X (already normalized)."""
        model = JumpRopeMLP().to("cpu")
        model.load_state_dict(
            torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            preds = (torch.sigmoid(logits) >= 0.5).float().numpy().flatten()
        return preds


# ---------------------------------------------------------------------------
# Pure NumPy fallback (for environments without PyTorch)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def train_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Path:
    """Minimal gradient-descent training with NumPy only."""
    mc = ModelConfig()
    rng = np.random.RandomState(config.seed)

    # Xavier init
    w1 = rng.randn(mc.input_dim, mc.hidden1).astype(np.float32) * np.sqrt(2.0 / mc.input_dim)
    b1 = np.zeros(mc.hidden1, dtype=np.float32)
    w2 = rng.randn(mc.hidden1, mc.hidden2).astype(np.float32) * np.sqrt(2.0 / mc.hidden1)
    b2 = np.zeros(mc.hidden2, dtype=np.float32)
    w3 = rng.randn(mc.hidden2, mc.output_dim).astype(np.float32) * np.sqrt(2.0 / mc.hidden2)
    b3 = np.zeros(mc.output_dim, dtype=np.float32)

    best_acc = 0.0
    n = len(X_train)

    for epoch in range(1, config.epochs + 1):
        # Mini-batch SGD
        perm = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = X_train[idx]
            yb = y_train[idx].reshape(-1, 1)

            # Forward
            h1 = _relu(xb @ w1 + b1)
            h2 = _relu(h1 @ w2 + b2)
            logits = h2 @ w3 + b3
            probs = _sigmoid(logits)

            # Backward (BCE gradient)
            eps = 1e-7
            d_logits = (probs - yb) / len(idx)
            d_w3 = h2.T @ d_logits
            d_b3 = d_logits.sum(axis=0)
            d_h2 = d_logits @ w3.T
            d_h2[h2 <= 0] = 0
            d_w2 = h1.T @ d_h2
            d_b2 = d_h2.sum(axis=0)
            d_h1 = d_h2 @ w2.T
            d_h1[h1 <= 0] = 0
            d_w1 = xb.T @ d_h1
            d_b1 = d_h1.sum(axis=0)

            # Weight decay
            lr = config.lr
            w3 -= lr * (d_w3 + config.weight_decay * w3)
            b3 -= lr * d_b3
            w2 -= lr * (d_w2 + config.weight_decay * w2)
            b2 -= lr * d_b2
            w1 -= lr * (d_w1 + config.weight_decay * w1)
            b1 -= lr * d_b1

        # Validate
        h1v = _relu(X_val @ w1 + b1)
        h2v = _relu(h1v @ w2 + b2)
        predsv = (_sigmoid(h2v @ w3 + b3) >= 0.5).astype(float).flatten()
        val_acc = (predsv == y_val).mean()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            config.output_dir.mkdir(parents=True, exist_ok=True)
            # Save raw weights in the binary format expected by C++
            weights = np.concatenate([
                w1.flatten(), b1.flatten(),
                w2.flatten(), b2.flatten(),
                w3.flatten(), b3.flatten(),
            ]).astype(np.float32)
            weights.tofile(str(config.output_dir / "jumprope_mlp.bin"))
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    print(f"Training complete. Best val_acc={best_acc:.4f}")
    return config.output_dir / "jumprope_mlp.bin"


def predict_numpy(bin_path: Path, X: np.ndarray) -> np.ndarray:
    """Forward pass using a raw-binary checkpoint (C++ layout)."""
    mc = ModelConfig()
    blob = np.fromfile(str(bin_path), dtype=np.float32)
    expected = mc.total_params
    if len(blob) != expected:
        raise ValueError(f"Weight file has {len(blob)} params, expected {expected}")
    o = 0
    w1 = blob[o : o + mc.input_dim * mc.hidden1].reshape(mc.input_dim, mc.hidden1); o += mc.input_dim * mc.hidden1
    b1 = blob[o : o + mc.hidden1]; o += mc.hidden1
    w2 = blob[o : o + mc.hidden1 * mc.hidden2].reshape(mc.hidden1, mc.hidden2); o += mc.hidden1 * mc.hidden2
    b2 = blob[o : o + mc.hidden2]; o += mc.hidden2
    w3 = blob[o : o + mc.hidden2 * mc.output_dim].reshape(mc.hidden2, mc.output_dim); o += mc.hidden2 * mc.output_dim
    b3 = blob[o : o + mc.output_dim]
    h1 = _relu(X @ w1 + b1)
    h2 = _relu(h1 @ w2 + b2)
    return (_sigmoid(h2 @ w3 + b3) >= 0.5).astype(float).flatten()


def predict(checkpoint_path: Path, X: np.ndarray, backend: str) -> np.ndarray:
    """Dispatcher for prediction that mirrors the training backend."""
    if backend == "torch":
        return predict_pytorch(checkpoint_path, X)
    if checkpoint_path.suffix == ".pt":
        return predict_pytorch(checkpoint_path, X)
    return predict_numpy(checkpoint_path, X)


# ---------------------------------------------------------------------------
# Export from PyTorch checkpoint to NCNN binary
# ---------------------------------------------------------------------------

def export_from_pytorch(checkpoint_path: Path, output_dir: Path) -> Path:
    """Load a PyTorch checkpoint and save raw weights for C++ inference."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available; use --numpy backend.")

    model = JumpRopeMLP()
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu", weights_only=True))
    model.eval()

    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    weights = []
    for layer in layers:
        weights.append(layer.weight.detach().numpy().T.flatten())  # transpose for C++ layout
        weights.append(layer.bias.detach().numpy().flatten())

    blob = np.concatenate(weights).astype(np.float32)
    expected = ModelConfig().total_params
    assert len(blob) == expected, f"Expected {expected} params, got {len(blob)}"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "jumprope_mlp.bin"
    blob.tofile(str(out_path))
    print(f"Exported {len(blob)} params to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JumpRope MLP classifier.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir,
                        help="Training data dir (CSV files).")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="Optional separate test data dir. If set, ALL samples "
                             "here are used as the test set and --data-dir is used "
                             "purely for training (no internal val split).")
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=["torch", "numpy", "auto"],
        default="auto",
        help="Training backend: torch (PyTorch), numpy (pure NumPy fallback), auto.",
    )
    parser.add_argument(
        "--export-only",
        type=Path,
        default=None,
        help="Skip training; export an existing .pt checkpoint to binary.",
    )
    return parser.parse_args(argv)


def _print_report(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    print(f"  samples : {metrics['total']}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}  recall: {metrics['recall']:.4f}  f1: {metrics['f1']:.4f}")
    print(f"  confusion: TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Export-only mode
    if args.export_only:
        export_from_pytorch(args.export_only, config.output_dir)
        return 0

    # Choose backend
    use_torch = args.backend == "torch" or (args.backend == "auto" and HAS_TORCH)
    backend_name = "PyTorch" if use_torch else "NumPy"
    print(f"Backend: {backend_name}")

    # ---- Load training data ----
    Xtr_raw, ytr_raw = load_csv_files(config.data_dir)
    Xtr = normalize_features(Xtr_raw)

    # ---- Build train/val splits (val slice is only for early-stopping) ----
    X_train, y_train, X_val, y_val = train_val_split(
        Xtr, ytr_raw, config.val_split, config.seed
    )
    if args.test_dir is not None:
        print(f"Train (augmented): {len(y_train)}, Val (augmented hold-out): {len(y_val)}")
    else:
        print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # ---- Train ----
    if use_torch:
        ckpt = train_pytorch(X_train, y_train, X_val, y_val, config)
        export_from_pytorch(ckpt, config.output_dir)
        predict_path = ckpt
    else:
        bin_path = train_numpy(X_train, y_train, X_val, y_val, config)
        predict_path = bin_path

    # ---- Evaluate on held-out test set (real data) ----
    if args.test_dir is not None:
        Xte_raw, yte = load_csv_files(args.test_dir)
        Xte = normalize_features(Xte_raw)
        print(f"\nTest set (real recordings): {len(yte)} samples "
              f"({int(yte.sum())} positive, {int(len(yte) - yte.sum())} negative)")
        preds = predict(predict_path, Xte, "torch" if use_torch else "numpy")
        metrics = evaluate_predictions(yte, preds)
        _print_report("TEST (real data, never seen in training)", metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
