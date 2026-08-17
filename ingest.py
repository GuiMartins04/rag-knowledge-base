"""Pipeline de ingestão: PDF -> chunks -> embeddings -> banco vetorial.

Uso:
    python ingest.py                      # usa data/rag-paper.pdf
    python ingest.py data/outro.pdf
"""

import sys
from pathlib import Path

from src.chunking import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.pdf_loader import load_pdf
from src.vector_store import get_collection, index_chunks


def ingest(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}")

    print(f"Lendo {path.name}...")
    pages = load_pdf(path)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(pages)} páginas -> {len(chunks)} chunks")

    print("Gerando embeddings e indexando (pode levar alguns segundos)...")
    index_chunks(chunks, source=path.name)

    print(f"Concluído. Total na coleção: {get_collection().count()} chunks")


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else "data/rag-paper.pdf")
