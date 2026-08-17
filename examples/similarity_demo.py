"""Script de teste: compara a similaridade entre embeddings de textos diferentes."""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

frases = [
    "O céu é azul durante o dia.",
    "Durante o dia, a cor do céu é azul.",
    "Eu gosto de comer pizza aos sábados.",
]


def gerar_vetor(texto: str) -> np.ndarray:
    resultado = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
    )
    return np.array(resultado.embeddings[0].values)


def similaridade_cosseno(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    return np.dot(vetor_a, vetor_b) / (np.linalg.norm(vetor_a) * np.linalg.norm(vetor_b))


vetores = [gerar_vetor(frase) for frase in frases]

print("Frases comparadas:")
for i, frase in enumerate(frases):
    print(f"  [{i}] {frase}")

print("\nSimilaridade entre [0] e [1] (frases parecidas):")
print(f"  {similaridade_cosseno(vetores[0], vetores[1]):.4f}")

print("\nSimilaridade entre [0] e [2] (frases diferentes):")
print(f"  {similaridade_cosseno(vetores[0], vetores[2]):.4f}")
