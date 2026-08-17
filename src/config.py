"""Configurações centrais do projeto."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# O SDK do Gemini emite um aviso sobre function calling em toda chamada de
# generate_content. Não se aplica ao nosso uso e só polui a saída.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY não encontrada. "
        "Copie .env.example para .env e preencha com sua chave."
    )

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Quantos chunks são recuperados para montar o contexto de cada resposta.
TOP_K = 4
