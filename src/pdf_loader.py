"""Leitura de PDFs, preservando o número da página de cada trecho de texto."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    """Uma página com texto extraída do PDF."""

    number: int
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    """Extrai o texto de cada página do PDF.

    Páginas sem texto (só imagem, por exemplo) são descartadas, porque não
    têm nada a contribuir para a busca.
    """
    reader = PdfReader(path)

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(number=number, text=text))

    return pages
