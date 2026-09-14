"""Common utilities for the Deep Learning final project.

The code is intentionally simple and explicit so it can be explained in an
academic report and during an oral defense.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def project_root() -> Path:
    """Return the project root whether the script is called from root or a subfolder."""
    return Path(__file__).resolve().parents[1]


def add_project_root_to_path() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU.

    On most ASUS VivoBook S14 configurations, this will usually return CPU.
    The subject asks us to verify device consistency, not necessarily to own a GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_classification_report(y_true: Iterable[int], y_pred: Iterable[int], target_names: List[str], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    path.write_text(report, encoding="utf-8")


def plot_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], target_names: List[str], path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: Dict[str, List[float]], path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valeur")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_named_parameters(model: torch.nn.Module, path: str | Path) -> None:
    """Save layer names and shapes for the report."""
    lines = []
    lines.append("Named parameters du modèle")
    lines.append("=" * 40)
    for name, parameter in model.named_parameters():
        lines.append(f"{name}: shape={tuple(parameter.shape)}, requires_grad={parameter.requires_grad}")
    lines.append("")
    lines.append("Clés du state_dict")
    lines.append("=" * 40)
    for key, tensor in model.state_dict().items():
        lines.append(f"{key}: shape={tuple(tensor.shape)}")
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
