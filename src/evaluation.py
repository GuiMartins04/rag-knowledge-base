"""Avaliação da qualidade do retrieval.

Mede se a busca vetorial recupera as páginas que realmente contêm a resposta.
O retrieval é a base do RAG: se o trecho certo não é recuperado, nenhum ajuste
de prompt corrige a resposta.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from src.embeddings import embed_queries
from src.vector_store import search_by_embedding


@dataclass
class EvalCase:
    """Uma pergunta e as páginas que contêm a resposta (gabarito)."""

    question: str
    expected_pages: list[int]


@dataclass
class CaseResult:
    """O resultado do retrieval para uma pergunta."""

    case: EvalCase
    retrieved_pages: list[int]
    rank: int | None
    """Posição (começando em 1) do primeiro acerto, ou None se nenhum."""

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """1º lugar vale 1,0; 2º vale 0,5; 3º vale 0,33; sem acerto vale 0."""
        return 1 / self.rank if self.rank else 0.0


@dataclass
class EvalReport:
    """Métricas agregadas de uma rodada de avaliação."""

    top_k: int
    results: list[CaseResult]

    @property
    def hit_rate(self) -> float:
        """Fração de perguntas em que alguma página correta foi recuperada."""
        return mean(float(result.hit) for result in self.results)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank: premia acertos nas primeiras posições."""
        return mean(result.reciprocal_rank for result in self.results)

    @property
    def misses(self) -> list[CaseResult]:
        return [result for result in self.results if not result.hit]


def load_dataset(path: str | Path = "evaluation/dataset.json") -> list[EvalCase]:
    """Carrega as perguntas e o gabarito do arquivo JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(question=case["question"], expected_pages=case["expected_pages"])
        for case in data["cases"]
    ]


def evaluate_case(case: EvalCase, embedding: list[float], top_k: int) -> CaseResult:
    """Busca no banco vetorial usando um embedding já calculado.

    Recebe o embedding pronto (em vez da pergunta em texto) para que o mesmo
    vetor seja reaproveitado ao comparar vários valores de top_k, evitando
    chamar a API de embeddings mais de uma vez para a mesma pergunta.
    """
    retrieved_pages = [
        result.page for result in search_by_embedding(embedding, top_k=top_k)
    ]

    rank = next(
        (
            position
            for position, page in enumerate(retrieved_pages, start=1)
            if page in case.expected_pages
        ),
        None,
    )

    return CaseResult(case=case, retrieved_pages=retrieved_pages, rank=rank)


def evaluate_all(cases: list[EvalCase], top_k_values: list[int]) -> list[EvalReport]:
    """Avalia o conjunto para vários valores de top_k.

    Os embeddings das perguntas são calculados uma única vez, em um lote só,
    e reaproveitados em cada valor de top_k comparado.
    """
    embeddings = embed_queries([case.question for case in cases])

    return [
        EvalReport(
            top_k=top_k,
            results=[
                evaluate_case(case, embedding, top_k)
                for case, embedding in zip(cases, embeddings)
            ],
        )
        for top_k in top_k_values
    ]
