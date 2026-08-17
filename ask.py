"""Faz uma pergunta ao RAG pela linha de comando.

Uso:
    python ask.py "O que é retrieval-augmented generation?"
"""

import sys

from src.rag import ask

sys.stdout.reconfigure(encoding="utf-8")

if len(sys.argv) < 2:
    raise SystemExit('Uso: python ask.py "sua pergunta aqui"')

question = " ".join(sys.argv[1:])
answer = ask(question)

print(f"PERGUNTA: {question}\n")
print("RESPOSTA:")
print(answer.text)

print("\nFONTES CONSULTADAS:")
for result in answer.sources:
    print(f"  - {result.source}, página {result.page} (distância {result.distance:.4f})")
