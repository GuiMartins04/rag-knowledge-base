"""Divisão do texto em chunks — os pedaços que viram embeddings."""

from dataclasses import dataclass

from src.pdf_loader import Page


@dataclass
class Chunk:
    """Um pedaço de texto pronto para ser indexado, com sua origem."""

    text: str
    page: int
    index: int


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Quebra o texto em pedaços de até `chunk_size` caracteres.

    Cada pedaço repete os últimos `overlap` caracteres do anterior. Essa
    sobreposição existe para que uma frase cortada no limite de um chunk
    ainda apareça inteira no chunk seguinte.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap precisa ser menor que chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap

    return chunks


def chunk_pages(
    pages: list[Page], chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    """Aplica o chunking página a página, guardando de qual página veio cada chunk.

    Chunks não atravessam a fronteira entre páginas: é uma simplificação que
    garante uma citação de página exata em cada resposta do RAG.
    """
    chunks: list[Chunk] = []

    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, page=page.number, index=len(chunks)))

    return chunks
