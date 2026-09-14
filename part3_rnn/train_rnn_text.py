"""Partie III - RNN, LSTM, GRU sur données textuelles.

Ce script compare trois architectures récurrentes sur une tâche de classification
textuelle réelle. Par défaut, il tente d'utiliser le corpus 20 Newsgroups de
scikit-learn avec deux catégories, car il est facile à traiter sans dépendances
complexes. Si le téléchargement échoue, un petit corpus intégré est utilisé
seulement comme fallback technique.

Objectifs couverts:
- tokenisation, vocabulaire, padding, mini-lots
- RNN simple, LSTM, GRU
- BPTT via loss.backward()
- gradient clipping
- comparaison stabilité/performance/coût de calcul
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
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

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_text_dataset(max_samples: int | None = None, download_20newsgroups: bool = False) -> Tuple[List[str], List[int], List[str]]:
    """Load a real text dataset if possible, otherwise a tiny fallback corpus."""
    try:
        categories = ["rec.autos", "sci.med"]
        data = fetch_20newsgroups(
            subset="all",
            categories=categories,
            remove=("headers", "footers", "quotes"),
            download_if_missing=download_20newsgroups,
        )
        texts = data.data
        labels = data.target.tolist()
        target_names = data.target_names
        if max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]
        return texts, labels, target_names
    except Exception as exc:  # pragma: no cover - fallback for offline execution
        print(f"WARNING: Could not download 20 Newsgroups ({exc}). Using tiny fallback corpus.")
        texts = [
            "the engine oil and tires must be checked before a long trip",
            "my car has a problem with the brake system and transmission",
            "the doctor prescribed a treatment after the medical examination",
            "the patient needs a diagnosis and regular health monitoring",
            "fuel consumption depends on speed vehicle weight and engine condition",
            "surgery requires sterile equipment and careful medical follow up",
            "the automobile dashboard shows a warning light about the battery",
            "blood pressure and heart rate are important clinical measurements",
        ] * 80
        labels = [0, 0, 1, 1, 0, 1, 0, 1] * 80
        target_names = ["auto", "medical"]
        return texts, labels, target_names


def build_vocab(texts: List[str], max_vocab: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def prepare_loaders(args):
    texts, labels, target_names = load_text_dataset(args.max_samples, args.download_20newsgroups)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    vocab = build_vocab(X_train, args.max_vocab)

    def tensorize(texts_part, labels_part):
        X = torch.tensor([encode_text(t, vocab, args.max_len) for t in texts_part], dtype=torch.long)
        y = torch.tensor(labels_part, dtype=torch.long)
        return TensorDataset(X, y)

    train_loader = DataLoader(tensorize(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tensorize(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(tensorize(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, vocab, target_names


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, rnn_type: str, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.rnn(embedded)
        if self.rnn_type == "lstm":
            _, (hidden, _) = output
        else:
            _, hidden = output
        final_hidden = hidden[-1]
        return self.fc(self.dropout(final_hidden))


def run_epoch(model, loader, criterion, optimizer, device, train: bool, clip: float):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    grad_norms = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                grad_norms.append(float(norm))
                optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += X_batch.size(0)
    avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
    return total_loss / total, correct / total, avg_grad


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(y_batch.numpy().tolist())
    return y_true, y_pred


def train_model(rnn_type, train_loader, val_loader, test_loader, vocab, target_names, args, device, out_dir):
    model = RecurrentClassifier(
        vocab_size=len(vocab),
        num_classes=len(target_names),
        rnn_type=rnn_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_idx=vocab[PAD_TOKEN],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "grad_norm": []}
    best_state, best_val_acc = None, -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, grad_norm = run_epoch(model, train_loader, criterion, optimizer, device, True, args.clip)
        val_loss, val_acc, _ = run_epoch(model, val_loader, criterion, optimizer, device, False, args.clip)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(grad_norm)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{rnn_type.upper()} epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | grad_norm={grad_norm:.4f}")

    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["best_val_accuracy"] = float(best_val_acc)
    metrics["trainable_parameters"] = int(count_parameters(model))
    metrics["device"] = str(device)

    torch.save(model.state_dict(), out_dir / "best_model_state_dict.pth")
    save_json(metrics, out_dir / "metrics.json")
    save_json(history, out_dir / "history.json")
    save_classification_report(y_true, y_pred, target_names, out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, target_names, out_dir / "confusion_matrix.png")
    plot_history(history, out_dir / "training_curves_and_gradients.png", f"Courbes {rnn_type.upper()}")
    describe_named_parameters(model, out_dir / "model_parameters_and_state_dict.txt")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download-20newsgroups", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, vocab, target_names = prepare_loaders(args)
    base_out = ensure_dir(ROOT / "results" / "part3_rnn")

    # Save vocabulary statistics for the report.
    vocab_info = {
        "vocab_size": len(vocab),
        "max_len": args.max_len,
        "pad_index": vocab[PAD_TOKEN],
        "unk_index": vocab[UNK_TOKEN],
        "target_names": target_names,
    }
    save_json(vocab_info, base_out / "vocabulary_info.json")

    rows = []
    for rnn_type in ["rnn", "lstm", "gru"]:
        print("\n" + "=" * 80)
        print(f"Experiment: {rnn_type.upper()}")
        out_dir = ensure_dir(base_out / rnn_type)
        metrics = train_model(rnn_type, train_loader, val_loader, test_loader, vocab, target_names, args, device, out_dir)
        rows.append({"model": rnn_type.upper(), **metrics})

    csv_path = base_out / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"experiments": rows}, base_out / "summary_metrics.json")
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()
