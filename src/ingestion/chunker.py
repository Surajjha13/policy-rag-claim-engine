"""Clause-aware, heading-aware chunking.

Two things make this "meaningful" rather than naive fixed-size splitting:

1. Clause boundaries: each page is split on numbered clause markers (e.g.
   "4.2 Waiting period..." or the exclusion list's "1. Pre-existing
   diseases") so a citation always points at one coherent clause, never a
   fragment glued to its neighbour. Pages with no such markers (cover
   page, definitions written as flowing prose) fall back to paragraph
   splitting.
2. Heading-aware sections: real headings ("DEFINITIONS", "SCOPE OF
   COVER", "WHAT WE EXCLUDE", ...) are detected in the raw text and each
   clause is tagged with the nearest heading that precedes it, carried
   across page boundaries. This matters because in the supplied policy
   the heading landing mid-page is the norm, not the exception (e.g. page
   7 contains the tail of the Critical Illness definitions AND the start
   of "SCOPE OF COVER").

A hard max length is enforced by sentence-boundary sub-splitting so no
chunk explodes the reranker's input size, without ever cutting a clause
anywhere except a sentence boundary.
"""

import re

from pydantic import BaseModel

from src.ingestion.pdf_parser import PageText
from src.ingestion.policy_sections import DEFAULT_SECTION, HEADING_MARKERS


class Chunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    section: str


CLAUSE_PATTERN = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,2})[.)]\s+")
MAX_CHUNK_CHARS = 900


def _split_clauses_with_offsets(text: str) -> list[tuple[int, str]]:
    """Return [(char_offset_in_page, clause_text), ...] in reading order."""
    matches = list(CLAUSE_PATTERN.finditer(text))
    if not matches:
        pieces = []
        cursor = 0
        for para in text.split("\n\n"):
            stripped = para.strip()
            if not stripped:
                cursor += len(para) + 2
                continue
            offset = text.index(stripped, cursor)
            pieces.append((offset, stripped))
            cursor = offset + len(stripped)
        return pieces or ([(0, text.strip())] if text.strip() else [])

    pieces = []
    leading = text[: matches[0].start()].strip()
    if leading:
        pieces.append((0, leading))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append((start, piece))
    return pieces


def _split_long(piece: str) -> list[str]:
    if len(piece) <= MAX_CHUNK_CHARS:
        return [piece]
    sentences = re.split(r"(?<=[.;])\s+", piece)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > MAX_CHUNK_CHARS and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf.strip())
    return out


def _headings_in_page(text: str) -> list[tuple[int, str]]:
    hits = [(text.find(marker), name) for marker, name in HEADING_MARKERS if marker in text]
    hits.sort(key=lambda h: h[0])
    return hits


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    current_section = DEFAULT_SECTION

    for page in pages:
        headings = _headings_in_page(page.text)
        for start_idx, clause in _split_clauses_with_offsets(page.text):
            while headings and headings[0][0] <= start_idx:
                current_section = headings.pop(0)[1]
            for sub in _split_long(clause):
                if len(sub) < 20:
                    continue
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk-{counter:04d}",
                        text=sub,
                        page=page.page,
                        section=current_section,
                    )
                )
        # any headings on this page that came after the last clause still
        # take effect for the next page's opening clauses.
        while headings:
            current_section = headings.pop(0)[1]

    return chunks
