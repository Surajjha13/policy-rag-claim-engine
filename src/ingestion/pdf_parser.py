import pymupdf as fitz
from pydantic import BaseModel


class PageText(BaseModel):
    page: int
    text: str


def extract_pages(pdf_path: str) -> list[PageText]:
    """Extract raw text per page, 1-indexed, preserving page numbers for citations."""
    doc = fitz.open(pdf_path)
    pages = [PageText(page=i + 1, text=doc[i].get_text("text")) for i in range(len(doc))]
    doc.close()
    return pages
