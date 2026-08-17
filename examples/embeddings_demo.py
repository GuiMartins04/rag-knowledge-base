"""Script de teste: gera o embedding de um texto usando a API do Gemini."""

import os

from dotenv import load_dotenv
from google import genai

# Lê o arquivo .env e carrega GEMINI_API_KEY na memória do processo
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

texto = "O céu é azul durante o dia."

resultado = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texto,
)

vetor = resultado.embeddings[0].values

print(f"Texto original: {texto}")
print(f"Tamanho do vetor: {len(vetor)} números")
print(f"Primeiros 5 valores: {vetor[:5]}")
