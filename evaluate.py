"""Avalia a qualidade do retrieval e compara diferentes valores de top_k.

Uso:
    python evaluate.py                  # avalia com top_k = 1, 3 e 5
    python evaluate.py --top-k 3        # avalia só com top_k = 3
    python evaluate.py --top-k 1 3 5 10 # compara vários
"""

import argparse
import sys

from src.evaluation import evaluate_all, load_dataset

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="Avaliação do retrieval do RAG.")
parser.add_argument(
    "--top-k",
    type=int,
    nargs="+",
    default=[1, 3, 5],
    help="Valores de top_k a comparar (padrão: 1 3 5)",
)
parser.add_argument(
    "--dataset",
    default="evaluation/dataset.json",
    help="Caminho do conjunto de avaliação",
)
args = parser.parse_args()

cases = load_dataset(args.dataset)
print(f"Conjunto de avaliação: {len(cases)} perguntas\n")

print("Gerando embeddings das perguntas em lote (1 chamada à API)...")
print(f"Comparando top_k = {args.top_k}...\n")
reports = evaluate_all(cases, args.top_k)

print("\n" + "=" * 46)
print(f"{'top_k':>6} | {'Hit Rate':>9} | {'MRR':>7} | {'Falhas':>7}")
print("-" * 46)
for report in reports:
    print(
        f"{report.top_k:>6} | {report.hit_rate:>8.1%} | "
        f"{report.mrr:>7.3f} | {len(report.misses):>7}"
    )
print("=" * 46)

# Detalha as falhas do maior top_k: são elas que mostram onde o retrieval erra.
worst = reports[-1]
if worst.misses:
    print(f"\nPerguntas sem acerto com top_k={worst.top_k}:\n")
    for result in worst.misses:
        print(f"  • {result.case.question}")
        print(
            f"    esperado: páginas {result.case.expected_pages} | "
            f"recuperado: páginas {result.retrieved_pages}\n"
        )
else:
    print(f"\nNenhuma falha com top_k={worst.top_k}.")
