"""Script de teste: lê o PDF, divide em chunks e mostra o resultado."""

import sys

from src.chunking import chunk_pages
from src.pdf_loader import load_pdf

# O terminal do Windows usa cp1252 por padrão e quebra ao imprimir símbolos
# matemáticos e ligaduras tipográficas comuns em papers acadêmicos.
sys.stdout.reconfigure(encoding="utf-8")

pages = load_pdf("data/rag-paper.pdf")
chunks = chunk_pages(pages)

print(f"Páginas com texto: {len(pages)}")
print(f"Chunks gerados: {len(chunks)}")
print(f"Tamanho médio: {sum(len(c.text) for c in chunks) // len(chunks)} caracteres\n")

print("=" * 70)
print("PRIMEIRO CHUNK (página 1):")
print("=" * 70)
print(chunks[0].text)

print("\n" + "=" * 70)
print("SOBREPOSIÇÃO — fim do chunk 0 vs início do chunk 1:")
print("=" * 70)
print(f"...{chunks[0].text[-200:]}")
print("\n--- chunk seguinte começa aqui ---\n")
print(f"{chunks[1].text[:200]}...")
