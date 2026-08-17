# RAG Knowledge Base

Sistema de perguntas e respostas sobre documentos PDF usando **RAG** (Retrieval-Augmented Generation): busca semântica em banco vetorial + geração de resposta por LLM, com **citação da página de origem** e recusa explícita quando a informação não está nos documentos.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF6B6B)
![Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4?logo=google&logoColor=white)

---

## O problema

Um LLM sozinho responde a partir do que memorizou durante o treinamento. Isso traz três limitações práticas:

- **Não conhece seus documentos** — contratos, manuais, políticas internas, papers específicos.
- **Alucina com confiança** — quando não sabe, tende a inventar uma resposta plausível.
- **Não é auditável** — não há como verificar de onde veio a informação.

RAG resolve os três: em vez de confiar na memória do modelo, o sistema **busca** os trechos relevantes nos documentos e entrega esses trechos ao LLM como contexto obrigatório da resposta.

---

## Como funciona

```mermaid
flowchart LR
    subgraph Indexação
        A[PDF] --> B[PyPDF<br/>extração por página]
        B --> C[Chunking<br/>1000 chars · overlap 200]
        C --> D[Embeddings<br/>RETRIEVAL_DOCUMENT]
        D --> E[(ChromaDB)]
    end

    subgraph Consulta
        F[Pergunta] --> G[Embedding<br/>RETRIEVAL_QUERY]
        G --> H[Similarity search<br/>top-k]
        E --> H
        H --> I[Contexto + prompt]
        I --> J[Gemini LLM]
        J --> K[Resposta + fontes]
    end
```

**Indexação** (uma vez por documento): o PDF é lido página a página, dividido em chunks com sobreposição, convertido em vetores pela API do Gemini e persistido no ChromaDB junto com o número da página de origem.

**Consulta** (a cada pergunta): a pergunta vira vetor, o ChromaDB devolve os chunks mais próximos, esses trechos são montados num prompt com regras rígidas de grounding, e o Gemini gera a resposta citando as páginas.

---

## Exemplo real

**Requisição**

```json
POST /query
{
  "question": "Quais são os dois tipos de memória que o RAG combina?",
  "top_k": 2
}
```

**Resposta**

```json
{
  "question": "Quais são os dois tipos de memória que o RAG combina?",
  "answer": "O RAG combina a memória paramétrica e a memória não paramétrica (página 9).",
  "sources": [
    {
      "source": "rag-paper.pdf",
      "page": 2,
      "distance": 0.5278,
      "excerpt": "We combine these components in a probabilistic model trained end-to-end..."
    },
    {
      "source": "rag-paper.pdf",
      "page": 9,
      "distance": 0.5514,
      "excerpt": "In this work, we presented hybrid generation models with access to parametric and non-parametric memory..."
    }
  ]
}
```

### Teste de alucinação

Perguntando algo que **não está** no documento indexado:

```bash
python ask.py "Qual é a capital da Austrália?"
```

```
RESPOSTA:
Não encontrei essa informação nos documentos.
```

O modelo sabe que a resposta é Canberra, mas o prompt o impede de usar conhecimento próprio. É este comportamento que torna o sistema confiável para uso corporativo.

---

## Decisões técnicas

**Chunking com sobreposição de 200 caracteres.** Sem sobreposição, uma frase que cai exatamente no corte entre dois chunks fica pela metade em ambos e não é recuperada por nenhum. A sobreposição garante que ideias na fronteira apareçam íntegras em pelo menos um chunk.

**`task_type` diferente para documento e pergunta.** O Gemini gera embeddings assimétricos: `RETRIEVAL_DOCUMENT` para o conteúdo indexado e `RETRIEVAL_QUERY` para a pergunta. Uma pergunta não se *parece* textualmente com sua resposta — ela *combina* com ela. Distinguir os dois papéis melhora a qualidade da recuperação.

**Chunks não atravessam páginas.** Simplificação deliberada: cada chunk pertence a exatamente uma página, o que permite citar a origem com precisão em vez de indicar um intervalo aproximado.

**Instrução de recusa explícita no prompt.** LLMs preenchem lacunas por padrão. Dar uma saída formal (`"Não encontrei essa informação nos documentos."`) é o que autoriza o modelo a admitir desconhecimento em vez de improvisar.

**Distância como sinal de qualidade.** A busca vetorial sempre retorna os *k* chunks menos distantes, mesmo quando nenhum é relevante. Na prática, perguntas cobertas pelo documento retornam distâncias entre 0,43 e 0,59; perguntas fora do escopo ficam acima de 0,89. Esse contraste é a base do filtro por limiar previsto no roadmap.

---

## Stack

| Componente | Tecnologia | Papel |
|---|---|---|
| Linguagem | Python 3.12 | — |
| API | FastAPI + Pydantic | Endpoints REST com validação e docs automáticas |
| Embeddings | `gemini-embedding-001` | Texto → vetor de 3072 dimensões |
| LLM | `gemini-3.6-flash` | Geração da resposta a partir do contexto |
| Banco vetorial | ChromaDB (persistente) | Armazenamento e similarity search |
| Leitura de PDF | PyPDF | Extração de texto por página |

---

## Como executar

**Pré-requisitos:** Python 3.12+ e uma API key do [Google AI Studio](https://aistudio.google.com/apikey).

```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/GuiMartins04/rag-knowledge-base.git
cd rag-knowledge-base

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate        # Linux/macOS

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Configurar a API key
cp .env.example .env             # depois preencha GEMINI_API_KEY

# 5. Indexar o documento de exemplo (o paper original de RAG, incluído no repo)
python ingest.py

# 6a. Perguntar pelo terminal
python ask.py "O que é retrieval-augmented generation?"

# 6b. Ou subir a API
uvicorn src.api:app --reload
```

Com a API no ar, a documentação interativa fica em **http://127.0.0.1:8000/docs**.

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e total de chunks indexados |
| `POST` | `/documents` | Upload de um PDF, que é indexado imediatamente |
| `POST` | `/query` | Pergunta sobre os documentos, com resposta e fontes |

---

## Estrutura do projeto

```
rag-knowledge-base/
├── src/
│   ├── config.py         # Configuração central (modelos, caminhos, parâmetros)
│   ├── pdf_loader.py     # Extração de texto do PDF, página a página
│   ├── chunking.py       # Divisão em chunks com sobreposição
│   ├── embeddings.py     # Geração de embeddings via Gemini (com batching)
│   ├── vector_store.py   # Indexação e busca no ChromaDB
│   ├── rag.py            # Montagem do contexto, prompt e geração da resposta
│   └── api.py            # Endpoints REST (FastAPI)
├── examples/             # Scripts didáticos de cada conceito, isolados
├── data/                 # Documentos de origem
├── ingest.py             # Pipeline de indexação (CLI)
└── ask.py                # Consulta pelo terminal (CLI)
```

A pasta `examples/` contém demonstrações isoladas de cada conceito — embeddings, similaridade de cosseno, chunking e busca vetorial — úteis para entender as peças separadamente:

```bash
python -m examples.similarity_demo
```

---

## Roadmap

- [ ] Filtro por limiar de distância, descartando contexto irrelevante antes de chamar o LLM
- [ ] Testes automatizados (pytest) para chunking e retrieval
- [ ] Suporte a mais formatos além de PDF (DOCX, Markdown, HTML)
- [ ] Interface web simples para upload e consulta
- [ ] Containerização com Docker
- [ ] Reranking dos chunks recuperados antes da geração
- [ ] Observabilidade: log de latência, tokens consumidos e distâncias por consulta

---

## Autor

**Guilherme Martins** — Analista de Automação e IA

[GitHub](https://github.com/GuiMartins04) · [LinkedIn](https://www.linkedin.com/in/guilherme-martins-06b972330)
