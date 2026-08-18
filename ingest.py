"""Pipeline de ingestão: PDF -> chunks -> embeddings -> banco vetorial.

Uso:
    python ingest.py                          # indexa data/rag-paper.pdf
    python ingest.py data/outro.pdf
    python ingest.py --reset                  # limpa a coleção antes de indexar
    python ingest.py --keep-bibliography      # não descarta páginas de referências
"""

import argparse
from pathlib import Path

from src.chunking import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.pdf_loader import load_pdf
from src.preprocessing import remove_bibliography
from src.vector_store import get_collection, index_chunks, reset_collection


def ingest(pdf_path: str, reset: bool = False, keep_bibliography: bool = False) -> None:
    path = Path(pdf_path)
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}")

    if reset:
        try:
            reset_collection()
            print("Coleção anterior removida.")
        except Exception:
            print("Nenhuma coleção anterior para remover.")

    print(f"Lendo {path.name}...")
    pages = load_pdf(path)

    if not keep_bibliography:
        pages, removed = remove_bibliography(pages)
        if removed:
            print(f"  Bibliografia descartada: páginas {removed}")

    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(pages)} páginas indexáveis -> {len(chunks)} chunks")

    print("Gerando embeddings e indexando (pode levar alguns segundos)...")
    index_chunks(chunks, source=path.name)

    print(f"Concluído. Total na coleção: {get_collection().count()} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa um PDF no banco vetorial.")
    parser.add_argument("pdf", nargs="?", default="data/rag-paper.pdf")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Apaga a coleção antes de indexar",
    )
    parser.add_argument(
        "--keep-bibliography",
        action="store_true",
        help="Mantém as páginas de referências na indexação",
    )
    args = parser.parse_args()

    ingest(args.pdf, reset=args.reset, keep_bibliography=args.keep_bibliography)
