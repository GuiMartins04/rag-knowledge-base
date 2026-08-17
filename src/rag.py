"""O RAG em si: recupera o contexto relevante e gera a resposta com o LLM."""

from dataclasses import dataclass

from google import genai

from src.config import GEMINI_API_KEY, LLM_MODEL, TOP_K
from src.vector_store import SearchResult, search

client = genai.Client(api_key=GEMINI_API_KEY)

# A instrução de recusa é o que segura a alucinação: sem ela, o modelo tende a
# completar a lacuna com conhecimento próprio quando o contexto não basta.
PROMPT_TEMPLATE = """Você responde perguntas usando APENAS os trechos de documento fornecidos abaixo.

Regras:
- Baseie a resposta exclusivamente no contexto fornecido, nunca em conhecimento próprio.
- Se o contexto não contiver a informação, responda exatamente: "Não encontrei essa informação nos documentos."
- Cite a página de onde veio cada informação, no formato (página X).
- Responda no mesmo idioma da pergunta.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


@dataclass
class Answer:
    """A resposta gerada junto dos trechos que a fundamentaram."""

    text: str
    sources: list[SearchResult]


def build_context(results: list[SearchResult]) -> str:
    """Concatena os trechos recuperados, cada um identificado por origem e página."""
    return "\n\n".join(
        f"[Trecho {position} — {result.source}, página {result.page}]\n{result.text}"
        for position, result in enumerate(results, start=1)
    )


def ask(question: str, top_k: int = TOP_K) -> Answer:
    """Executa o ciclo completo do RAG para uma pergunta."""
    results = search(question, top_k=top_k)

    if not results:
        return Answer(
            text="Não encontrei essa informação nos documentos.",
            sources=[],
        )

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=PROMPT_TEMPLATE.format(
            context=build_context(results),
            question=question,
        ),
    )

    return Answer(text=(response.text or "").strip(), sources=results)
