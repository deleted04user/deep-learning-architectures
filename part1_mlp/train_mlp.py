"""Partie I - MLP et ingénierie PyTorch.

Dataset: Breast Cancer Wisconsin, disponible dans scikit-learn.
Objectifs couverts:
- nn.Sequential vs classe personnalisée nn.Module
- inspection de named_parameters() et state_dict()
- initialisations: gaussienne, constante, Xavier
- sauvegarde/rechargement du meilleur modèle
- exécution sur le device disponible CPU/GPU
- métriques: accuracy, precision, recall, F1-score, matrice de confusion
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    compute_classification_metrics,
    count_parameters,
    describe_named_parameters,
    ensure_dir,
    get_device,
    plot_confusion_matrix,
    plot_history,
    save_classification_report,
    save_json,
    set_seed,
)


class CustomMLP(nn.Module):
    """MLP implemented as a custom nn.Module class."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def build_sequential_mlp(input_dim: int, hidden1: int = 64, hidden2: int = 32, output_dim: int = 2) -> nn.Sequential:
    """MLP implemented with nn.Sequential."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, output_dim),
    )


def initialize_weights(model: nn.Module, strategy: str) -> None:
    """Apply one requested initialization strategy to all Linear layers."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "gaussian":
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                nn.init.zeros_(module.bias)
            elif strategy == "constant":
                # A small positive constant is intentionally weak: it breaks symmetry poorly.
                nn.init.constant_(module.weight, 0.01)
                nn.init.zeros_(module.bias)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")


def prepare_data(batch_size: int):
    dataset = load_breast_cancer()
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)

    # Split: 60% train, 20% validation, 20% test.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # Fit scaler only on training data to avoid data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def make_loader(X_arr, y_arr, shuffle: bool) -> DataLoader:
        tensors = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    return (
        make_loader(X_train, y_train, True),
        make_loader(X_val, y_val, False),
        make_loader(X_test, y_test, False),
        dataset.feature_names.tolist(),
        dataset.target_names.tolist(),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy().tolist())
    return y_true, y_pred


def train_one_model(model: nn.Module, train_loader, val_loader, test_loader, args, device, out_dir: Path, target_names):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Training failed: best_state is None")

    # Save and reload the best model to prove the state_dict workflow.
    model_path = out_dir / "best_model_state_dict.pth"
    torch.save(best_state, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["best_val_accuracy"] = float(best_val_acc)
    metrics["trainable_parameters"] = int(count_parameters(model))
    metrics["device"] = str(device)

    save_json(metrics, out_dir / "metrics.json")
    save_json(history, out_dir / "history.json")
    save_classification_report(y_true, y_pred, target_names, out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, target_names, out_dir / "confusion_matrix.png")
    plot_history(history, out_dir / "training_curves.png", title="Courbes apprentissage/validation")
    describe_named_parameters(model, out_dir / "model_parameters_and_state_dict.txt")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, feature_names, target_names = prepare_data(args.batch_size)
    input_dim = len(feature_names)

    base_out = ensure_dir(ROOT / "results" / "part1_mlp")
    experiment_rows = []

    model_builders = {
        "sequential": lambda: build_sequential_mlp(input_dim),
        "custom_class": lambda: CustomMLP(input_dim),
    }
    init_strategies = ["gaussian", "constant", "xavier"]

    for model_name, builder in model_builders.items():
        for init in init_strategies:
            print("\n" + "=" * 80)
            print(f"Experiment: {model_name} | initialization={init}")
            out_dir = ensure_dir(base_out / f"{model_name}_{init}")
            model = builder()
            initialize_weights(model, init)
            metrics = train_one_model(model, train_loader, val_loader, test_loader, args, device, out_dir, target_names)
            row = {"model": model_name, "initialization": init, **metrics}
            experiment_rows.append(row)

    csv_path = base_out / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(experiment_rows)

    save_json({"experiments": experiment_rows}, base_out / "summary_metrics.json")
    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    main()
