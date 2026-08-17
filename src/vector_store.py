"""Armazenamento e busca dos chunks no banco vetorial (ChromaDB)."""

from dataclasses import dataclass

import chromadb

from src.chunking import Chunk
from src.config import CHROMA_PATH, COLLECTION_NAME, TOP_K
from src.embeddings import embed_documents, embed_query


@dataclass
class SearchResult:
    """Um chunk recuperado do banco, com sua origem e proximidade."""

    text: str
    source: str
    page: int
    distance: float


def get_collection():
    """Abre (ou cria) a coleção persistida em disco."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_chunks(chunks: list[Chunk], source: str) -> None:
    """Gera os embeddings dos chunks e grava tudo na coleção.

    O id combina arquivo e posição, então reindexar o mesmo PDF atualiza os
    registros existentes em vez de duplicá-los.
    """
    collection = get_collection()
    texts = [chunk.text for chunk in chunks]

    collection.upsert(
        ids=[f"{source}::{chunk.index}" for chunk in chunks],
        embeddings=embed_documents(texts),
        documents=texts,
        metadatas=[{"source": source, "page": chunk.page} for chunk in chunks],
    )


def search(question: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Recupera os chunks mais parecidos com a pergunta."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=top_k,
    )

    return [
        SearchResult(
            text=text,
            source=str(metadata["source"]),
            page=int(metadata["page"]),
            distance=distance,
        )
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
