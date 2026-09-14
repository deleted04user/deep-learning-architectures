"""Mini système Seq2Seq encodeur-décodeur récurrent.

Cette démonstration couvre:
- vocabulaire source/cible
- tokens spéciaux <sos>, <eos>, <pad>, <unk>
- encodeur GRU
- décodeur GRU
- teacher forcing
- décodage glouton
- beam search
- score BLEU simplifié

Pour un rendu final plus solide, remplacer les paires intégrées par un corpus parallèle
réel comme Tatoeba fra-eng simplifié.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ensure_dir, get_device, save_json, set_seed  # noqa: E402

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

PAIRS = [
    ("i am a student", "je suis un etudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she likes music", "elle aime la musique"),
    ("we study deep learning", "nous etudions le deep learning"),
    ("the model is training", "le modele est en entrainement"),
    ("the data is clean", "les donnees sont propres"),
    ("i like artificial intelligence", "j aime l intelligence artificielle"),
    ("this result is good", "ce resultat est bon"),
    ("the network learns features", "le reseau apprend des caracteristiques"),
    ("we compare models", "nous comparons des modeles"),
    ("the sequence is short", "la sequence est courte"),
    ("the decoder predicts words", "le decodeur predit des mots"),
] * 20


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(sentences: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))
    for word, _ in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(sentence: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(sentence)]
    ids = ids[: max_len - 2]
    ids = [vocab[SOS]] + ids + [vocab[EOS]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in vocab.items()}


def prepare_data(max_len: int, batch_size: int):
    src_sentences = [p[0] for p in PAIRS]
    tgt_sentences = [p[1] for p in PAIRS]
    src_vocab = build_vocab(src_sentences)
    tgt_vocab = build_vocab(tgt_sentences)
    src_tensor = torch.tensor([encode(s, src_vocab, max_len) for s in src_sentences], dtype=torch.long)
    tgt_tensor = torch.tensor([encode(s, tgt_vocab, max_len) for s in tgt_sentences], dtype=torch.long)
    dataset = TensorDataset(src_tensor, tgt_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, src_vocab, tgt_vocab


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token.unsqueeze(1))
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, tgt_sos_idx: int, tgt_eos_idx: int, max_len: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.max_len = max_len

    def forward(self, src, tgt=None, teacher_forcing_ratio: float = 0.5):
        batch_size = src.size(0)
        _, hidden = self.encoder(src)
        input_token = torch.full((batch_size,), self.tgt_sos_idx, dtype=torch.long, device=src.device)
        logits_steps = []
        for t in range(1, self.max_len):
            logits, hidden = self.decoder(input_token, hidden)
            logits_steps.append(logits.unsqueeze(1))
            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = logits.argmax(dim=1)
        return torch.cat(logits_steps, dim=1)


def train(model, loader, tgt_pad_idx, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    history = {"loss": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt, teacher_forcing_ratio=args.teacher_forcing)
            # Predict target positions 1..max_len-1
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * src.size(0)
            total_tokens += src.size(0)
        avg_loss = total_loss / total_tokens
        history["loss"].append(avg_loss)
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:04d}/{args.epochs} | loss={avg_loss:.4f}")
    return history


def decode_tokens(ids: List[int], inv_vocab: Dict[int, str]) -> List[str]:
    tokens = []
    for idx in ids:
        word = inv_vocab.get(idx, UNK)
        if word in {SOS, PAD}:
            continue
        if word == EOS:
            break
        tokens.append(word)
    return tokens


def greedy_decode(model, sentence: str, src_vocab, tgt_vocab, max_len, device) -> List[str]:
    inv_tgt = invert_vocab(tgt_vocab)
    src = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt=None, teacher_forcing_ratio=0.0)
        ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return decode_tokens(ids, inv_tgt)


def beam_search_decode(model, sentence: str, src_vocab, tgt_vocab, max_len, device, beam_width: int = 3) -> List[str]:
    inv_tgt = invert_vocab(tgt_vocab)
    src = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        _, hidden = model.encoder(src)
        beams = [([tgt_vocab[SOS]], 0.0, hidden)]
        for _ in range(max_len - 1):
            new_beams = []
            for seq, score, h in beams:
                if seq[-1] == tgt_vocab[EOS]:
                    new_beams.append((seq, score, h))
                    continue
                input_token = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                logits, new_hidden = model.decoder(input_token, h)
                log_probs = F.log_softmax(logits, dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_width, dim=-1)
                for token_score, token_id in zip(top_scores[0], top_ids[0]):
                    new_seq = seq + [int(token_id.item())]
                    new_score = score + float(token_score.item())
                    new_beams.append((new_seq, new_score, new_hidden))
            beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
        best_seq = beams[0][0]
    return decode_tokens(best_seq, inv_tgt)


def bleu_unigram(reference: List[str], candidate: List[str]) -> float:
    if not candidate:
        return 0.0
    ref_counter = Counter(reference)
    cand_counter = Counter(candidate)
    overlap = sum(min(count, ref_counter[token]) for token, count in cand_counter.items())
    precision = overlap / len(candidate)
    brevity_penalty = 1.0 if len(candidate) > len(reference) else math.exp(1 - len(reference) / max(1, len(candidate)))
    return brevity_penalty * precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(ROOT / "results" / "part3_seq2seq")
    loader, src_vocab, tgt_vocab = prepare_data(args.max_len, args.batch_size)

    encoder = Encoder(len(src_vocab), args.embedding_dim, args.hidden_dim, src_vocab[PAD])
    decoder = Decoder(len(tgt_vocab), args.embedding_dim, args.hidden_dim, tgt_vocab[PAD])
    model = Seq2Seq(encoder, decoder, tgt_vocab[SOS], tgt_vocab[EOS], args.max_len).to(device)

    history = train(model, loader, tgt_vocab[PAD], args, device)
    torch.save(model.state_dict(), out_dir / "seq2seq_gru_state_dict.pth")

    examples = []
    for src_sentence, tgt_sentence in PAIRS[:8]:
        greedy = greedy_decode(model, src_sentence, src_vocab, tgt_vocab, args.max_len, device)
        beam = beam_search_decode(model, src_sentence, src_vocab, tgt_vocab, args.max_len, device, beam_width=3)
        reference = tokenize(tgt_sentence)
        examples.append(
            {
                "source": src_sentence,
                "reference": tgt_sentence,
                "greedy": " ".join(greedy),
                "beam_search": " ".join(beam),
                "bleu1_greedy": bleu_unigram(reference, greedy),
                "bleu1_beam": bleu_unigram(reference, beam),
            }
        )

    save_json({"history": history, "examples": examples}, out_dir / "seq2seq_results.json")
    print(f"Saved results to {out_dir / 'seq2seq_results.json'}")


if __name__ == "__main__":
    main()
