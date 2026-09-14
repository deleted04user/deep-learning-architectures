"""Partie II - CNN et vision par ordinateur.

Objectifs couverts:
- justification CNN vs MLP
- calculs et implémentations manuelles: corrélation croisée 2D, max-pooling, average-pooling
- comparaison avec les couches PyTorch
- CNN inspiré de LeNet
- expériences: padding, stride, pooling, nombre de filtres, convolution 1x1
- visualisation de cartes de caractéristiques
- comparaison MLP vs CNN sur le même dataset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

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


def output_size(w: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    return (w - kernel + 2 * padding) // stride + 1


def xcorr2d_manual(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Manual 2D cross-correlation without padding and stride=1.

    X: H x W, K: kH x kW
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


def max_pool2d_manual(X: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    p_h, p_w = pool_size
    out_h = X.shape[0] // p_h
    out_w = X.shape[1] // p_w
    Y = torch.zeros((out_h, out_w), dtype=X.dtype)
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = X[i * p_h : (i + 1) * p_h, j * p_w : (j + 1) * p_w].max()
    return Y


def avg_pool2d_manual(X: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    p_h, p_w = pool_size
    out_h = X.shape[0] // p_h
    out_w = X.shape[1] // p_w
    Y = torch.zeros((out_h, out_w), dtype=X.dtype)
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = X[i * p_h : (i + 1) * p_h, j * p_w : (j + 1) * p_w].mean()
    return Y


def verify_manual_ops(out_dir: Path) -> None:
    """Compare manual operations with PyTorch operations on a small tensor."""
    X = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    K = torch.tensor([[0., 1.], [2., 3.]])

    manual_corr = xcorr2d_manual(X, K)
    torch_corr = F.conv2d(X.view(1, 1, 3, 3), K.view(1, 1, 2, 2)).view(2, 2)

    manual_max = max_pool2d_manual(X[:2, :2], (2, 2))
    torch_max = F.max_pool2d(X[:2, :2].view(1, 1, 2, 2), kernel_size=2).view(1, 1)

    manual_avg = avg_pool2d_manual(X[:2, :2], (2, 2))
    torch_avg = F.avg_pool2d(X[:2, :2].view(1, 1, 2, 2), kernel_size=2).view(1, 1)

    lines = [
        "Vérification des opérations manuelles",
        "=" * 45,
        f"X =\n{X}",
        f"K =\n{K}",
        f"Corrélation manuelle =\n{manual_corr}",
        f"Corrélation PyTorch =\n{torch_corr}",
        f"Max-pooling manuel = {manual_max.item():.4f}",
        f"Max-pooling PyTorch = {torch_max.item():.4f}",
        f"Average-pooling manuel = {manual_avg.item():.4f}",
        f"Average-pooling PyTorch = {torch_avg.item():.4f}",
        "",
        "Exemple de calcul dimensionnel Fashion-MNIST 28x28, kernel=3, stride=1, padding=0:",
        f"Sortie convolution = floor((28 - 3 + 2*0)/1) + 1 = {output_size(28, 3, 0, 1)}",
        f"Après pooling 2x2 = {output_size(26, 2, 0, 2)}",
    ]
    ensure_dir(out_dir)
    (out_dir / "manual_operations_verification.txt").write_text("\n".join(lines), encoding="utf-8")


class MLPImageBaseline(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeNetVariant(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        num_classes: int,
        filters1: int = 32,
        filters2: int = 64,
        padding: int = 0,
        stride: int = 1,
        pooling: str = "max",
        use_conv1x1: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=3, stride=stride, padding=padding)
        self.conv1x1 = nn.Conv2d(filters1, filters1, kernel_size=1) if use_conv1x1 else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=3, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
        self.relu = nn.ReLU()

        # Infer flatten size with a dummy tensor.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flatten_size = self._features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _features(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.conv1x1(x)
        x = self.relu(self.conv2(x))
        if x.shape[-2] >= 2 and x.shape[-1] >= 2:
            x = self.pool(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def load_fashion_mnist(batch_size: int, max_train: int | None, max_test: int | None):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root=str(ROOT / "data"), train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=str(ROOT / "data"), train=False, download=True, transform=transform)

    if max_train:
        train_dataset = Subset(train_dataset, list(range(min(max_train, len(train_dataset)))))
    if max_test:
        test_dataset = Subset(test_dataset, list(range(min(max_test, len(test_dataset)))))

    val_size = max(1, int(0.15 * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return train_loader, val_loader, test_loader, 28, 1, 10, class_names


def load_digits_dataset(batch_size: int):
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    X = X[:, None, :, :]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    def loader(X_arr, y_arr, shuffle):
        ds = TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    class_names = [str(i) for i in range(10)]
    return loader(X_train, y_train, True), loader(X_val, y_val, False), loader(X_test, y_test, False), 8, 1, 10, class_names


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(y_batch.numpy().tolist())
    return y_true, y_pred


def visualize_feature_maps(model: LeNetVariant, loader: DataLoader, device, out_dir: Path) -> None:
    model.eval()
    X_batch, _ = next(iter(loader))
    image = X_batch[:1].to(device)
    with torch.no_grad():
        feature_maps = model.relu(model.conv1(image)).cpu()[0]

    n_maps = min(8, feature_maps.shape[0])
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n_maps, figsize=(2 * n_maps, 2))
    if n_maps == 1:
        axes = [axes]
    for i in range(n_maps):
        axes[i].imshow(feature_maps[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Map {i+1}")
    fig.suptitle("Exemples de cartes de caractéristiques - première convolution")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_maps_conv1.png", dpi=150)
    plt.close(fig)


def train_model(model, train_loader, val_loader, test_loader, args, device, out_dir, class_names):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state, best_val_acc = None, -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["best_val_accuracy"] = float(best_val_acc)
    metrics["trainable_parameters"] = int(count_parameters(model))
    metrics["device"] = str(device)
    torch.save(model.state_dict(), out_dir / "best_model_state_dict.pth")
    save_json(metrics, out_dir / "metrics.json")
    save_json(history, out_dir / "history.json")
    save_classification_report(y_true, y_pred, class_names, out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
    plot_history(history, out_dir / "training_curves.png", "Courbes CNN/MLP")
    describe_named_parameters(model, out_dir / "model_parameters_and_state_dict.txt")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fashionmnist", "digits"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    base_out = ensure_dir(ROOT / "results" / "part2_cnn")
    verify_manual_ops(base_out)

    if args.dataset == "fashionmnist":
        train_loader, val_loader, test_loader, image_size, in_channels, num_classes, class_names = load_fashion_mnist(
            args.batch_size, args.max_train, args.max_test
        )
    else:
        train_loader, val_loader, test_loader, image_size, in_channels, num_classes, class_names = load_digits_dataset(args.batch_size)

    experiments: Dict[str, nn.Module] = {
        "mlp_baseline": MLPImageBaseline(image_size * image_size * in_channels, num_classes),
        "cnn_baseline_maxpool": LeNetVariant(image_size, in_channels, num_classes, pooling="max"),
        "cnn_average_pooling": LeNetVariant(image_size, in_channels, num_classes, pooling="avg"),
        "cnn_with_padding": LeNetVariant(image_size, in_channels, num_classes, padding=1, pooling="max"),
        "cnn_fewer_filters": LeNetVariant(image_size, in_channels, num_classes, filters1=16, filters2=32, pooling="max"),
        "cnn_with_1x1_conv": LeNetVariant(image_size, in_channels, num_classes, padding=1, use_conv1x1=True, pooling="max"),
    }

    rows = []
    for name, model in experiments.items():
        print("\n" + "=" * 80)
        print(f"Experiment: {name}")
        out_dir = ensure_dir(base_out / name)
        metrics = train_model(model, train_loader, val_loader, test_loader, args, device, out_dir, class_names)
        rows.append({"experiment": name, **metrics})
        if isinstance(model, LeNetVariant) and name == "cnn_baseline_maxpool":
            visualize_feature_maps(model, test_loader, device, out_dir)

    csv_path = base_out / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"experiments": rows}, base_out / "summary_metrics.json")
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()
