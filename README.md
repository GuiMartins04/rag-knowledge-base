# RAG Knowledge Base

A Retrieval-Augmented Generation (RAG) application for querying documents using embeddings, vector search and LLMs.

> 🚧 Projeto em construção — documentação completa será adicionada ao final do desenvolvimento.

## Stack

- Python
- FastAPI
- Google Gemini API (embeddings + geração de texto)
- ChromaDB (banco vetorial)
- PyPDF (leitura de PDF)

## Como rodar localmente

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
cp .env.example .env          # depois preencha sua GEMINI_API_KEY
```
