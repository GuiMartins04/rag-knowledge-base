"""Script de teste: armazena textos no ChromaDB e faz busca por similaridade."""

import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# PersistentClient grava em disco (pasta chroma_db/), diferente do Client() em memória
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# get_or_create evita erro se a coleção já existir de uma execução anterior
colecao = client_chroma.get_or_create_collection(name="teste")

documentos = [
    "O céu é azul durante o dia por causa da dispersão da luz solar.",
    "A pizza margherita leva tomate, mussarela e manjericão.",
    "As nuvens são formadas por gotículas de água suspensas na atmosfera.",
    "O futebol brasileiro tem cinco títulos mundiais.",
]


def gerar_vetores(textos: list[str]) -> list[list[float]]:
    resultado = client_gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=textos,
    )
    return [embedding.values for embedding in resultado.embeddings]


# Indexação: guardamos o vetor, o texto original e um id para cada documento
print("Gerando embeddings e salvando no ChromaDB...")
colecao.upsert(
    ids=[f"doc_{i}" for i in range(len(documentos))],
    embeddings=gerar_vetores(documentos),
    documents=documentos,
)
print(f"Total de documentos na coleção: {colecao.count()}\n")

# Busca: a pergunta também vira vetor, e o Chroma acha os mais próximos
pergunta = "Por que o céu tem essa cor?"
vetor_pergunta = gerar_vetores([pergunta])[0]

resultados = colecao.query(query_embeddings=[vetor_pergunta], n_results=2)

print(f"Pergunta: {pergunta}\n")
print("Documentos mais relevantes encontrados:")
for texto, distancia in zip(resultados["documents"][0], resultados["distances"][0]):
    print(f"  [distância: {distancia:.4f}] {texto}")
