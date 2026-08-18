"""Limpeza do documento antes da indexação.

Páginas de bibliografia são ruído para busca semântica: são listas de nomes,
instituições e títulos que competem com o conteúdo real. Numa pergunta como
"quem são os autores?", as referências vencem a capa do paper — foi
exatamente a falha observada na avaliação do retrieval.
"""

import re

from src.pdf_loader import Page

# Uma entrada de bibliografia começa com o marcador da citação: "[42] Autor..."
REFERENCE_LINE = re.compile(r"^\[\d{1,3}\]")

# Limiar medido no paper de referência: páginas de conteúdo ficam entre 0% e
# 12% de linhas nesse formato, enquanto páginas de bibliografia ficam entre
# 19% e 25%. 15% cai na folga entre os dois grupos.
BIBLIOGRAPHY_RATIO = 0.15
MIN_REFERENCE_LINES = 3


def is_bibliography_page(text: str) -> bool:
    """Indica se a página é majoritariamente uma lista de referências."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    reference_lines = sum(1 for line in lines if REFERENCE_LINE.match(line))
    if reference_lines < MIN_REFERENCE_LINES:
        return False

    return reference_lines / len(lines) >= BIBLIOGRAPHY_RATIO


def remove_bibliography(pages: list[Page]) -> tuple[list[Page], list[int]]:
    """Separa as páginas de conteúdo das de bibliografia.

    Retorna as páginas mantidas e os números das páginas descartadas.
    """
    kept = [page for page in pages if not is_bibliography_page(page.text)]
    removed = [page.number for page in pages if is_bibliography_page(page.text)]
    return kept, removed
