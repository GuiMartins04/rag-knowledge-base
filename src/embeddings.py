"""Geração de embeddings através da API do Gemini."""

from google import genai
from google.genai import types

from src.config import EMBEDDING_MODEL, GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

# A API limita quantos textos cabem numa chamada; 50 é um lote conservador.
BATCH_SIZE = 50


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(embedding.values for embedding in response.embeddings)

    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeddings dos trechos que serão armazenados no banco."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embedding de uma pergunta feita pelo usuário."""
    return _embed([text], "RETRIEVAL_QUERY")[0]
