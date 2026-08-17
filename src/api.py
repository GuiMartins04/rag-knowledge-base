"""API REST do RAG."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.chunking import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from src.pdf_loader import load_pdf
from src.rag import ask
from src.vector_store import get_collection, index_chunks

DATA_DIR = Path("data")

app = FastAPI(
    title="RAG Knowledge Base",
    description="Perguntas e respostas sobre documentos PDF, com citação das fontes.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        examples=["O que é retrieval-augmented generation?"],
    )
    top_k: int = Field(default=TOP_K, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int
    distance: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    total_in_collection: int


@app.get("/health")
def health() -> dict:
    """Verifica se a API está de pé e quantos chunks estão indexados."""
    return {"status": "ok", "indexed_chunks": get_collection().count()}


@app.post("/documents", response_model=IngestResponse)
def upload_document(file: UploadFile) -> IngestResponse:
    """Recebe um PDF, indexa seu conteúdo e o deixa disponível para consulta."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    DATA_DIR.mkdir(exist_ok=True)

    # Path(...).name descarta qualquer diretório vindo no nome enviado, evitando
    # que um upload malicioso escreva fora da pasta data/.
    destination = DATA_DIR / Path(file.filename).name
    destination.write_bytes(file.file.read())

    pages = load_pdf(destination)
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="Não foi possível extrair texto deste PDF.",
        )

    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    index_chunks(chunks, source=destination.name)

    return IngestResponse(
        filename=destination.name,
        pages=len(pages),
        chunks=len(chunks),
        total_in_collection=get_collection().count(),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Responde uma pergunta com base nos documentos indexados."""
    answer = ask(request.question, top_k=request.top_k)

    return QueryResponse(
        question=request.question,
        answer=answer.text,
        sources=[
            Source(
                source=result.source,
                page=result.page,
                distance=result.distance,
                excerpt=result.text[:200].strip() + "...",
            )
            for result in answer.sources
        ],
    )
