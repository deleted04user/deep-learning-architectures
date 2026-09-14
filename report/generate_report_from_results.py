"""Generate a Markdown report section from produced experiment metrics.

Usage after running experiments:
    python report/generate_report_from_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "rapport_final_resultats.md"


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows):
    if not rows:
        return "_Résultats non disponibles. Exécuter le script correspondant._\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for h in headers:
            v = row.get(h, "")
            try:
                fv = float(v)
                if 0 <= fv <= 1:
                    v = f"{fv:.4f}"
                else:
                    v = f"{fv:.2f}"
            except Exception:
                pass
            values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    part1 = read_csv(RESULTS / "part1_mlp" / "summary_metrics.csv")
    part2 = read_csv(RESULTS / "part2_cnn" / "summary_metrics.csv")
    part3 = read_csv(RESULTS / "part3_rnn" / "summary_metrics.csv")

    text = []
    text.append("# Sections de résultats générées automatiquement\n")
    text.append("Ces tableaux doivent être copiés dans le rapport Word final après l'exécution des scripts.\n")

    text.append("## Partie I - MLP / données tabulaires\n")
    text.append(md_table(part1))
    text.append("Analyse attendue : comparer nn.Sequential et classe personnalisée, puis expliquer l'impact des initialisations gaussienne, constante et Xavier sur la stabilité de l'apprentissage.\n")

    text.append("## Partie II - CNN / images\n")
    text.append(md_table(part2))
    text.append("Analyse attendue : comparer MLP et CNN, puis interpréter l'effet du padding, du pooling, du nombre de filtres et de la convolution 1x1.\n")

    text.append("## Partie III - RNN / LSTM / GRU\n")
    text.append(md_table(part3))
    text.append("Analyse attendue : discuter stabilité, capacité mémoire, gradient clipping, coût de calcul et performance relative de RNN, LSTM et GRU.\n")

    seq_path = RESULTS / "part3_seq2seq" / "seq2seq_results.json"
    text.append("## Mini Seq2Seq\n")
    if seq_path.exists():
        data = json.loads(seq_path.read_text(encoding="utf-8"))
        examples = data.get("examples", [])
        rows = [
            {
                "source": e["source"],
                "reference": e["reference"],
                "greedy": e["greedy"],
                "beam_search": e["beam_search"],
                "bleu1_greedy": e["bleu1_greedy"],
                "bleu1_beam": e["bleu1_beam"],
            }
            for e in examples
        ]
        text.append(md_table(rows))
    else:
        text.append("_Résultats Seq2Seq non disponibles. Exécuter `python part3_rnn/tiny_seq2seq.py`._\n")

    OUT.write_text("\n".join(text), encoding="utf-8")
    print(f"Generated {OUT}")


if __name__ == "__main__":
    main()
