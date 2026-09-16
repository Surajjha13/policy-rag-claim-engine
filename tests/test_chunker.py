from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_parser import PageText


def test_chunker_splits_on_numbered_clauses_and_keeps_metadata():
    page = PageText(
        page=10,
        text=(
            "4.1 Waiting period for pre-existing diseases is 48 months.\n\n"
            "4.2 Initial waiting period is 30 days from policy inception.\n"
        ),
    )
    chunks = chunk_pages([page])
    assert len(chunks) == 2
    assert chunks[0].page == 10
    assert "48 months" in chunks[0].text
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunker_never_produces_chunk_over_max_chars():
    long_text = "5.1 " + ("Sub-limit clause text. " * 200)
    page = PageText(page=17, text=long_text)
    chunks = chunk_pages([page])
    assert all(len(c.text) <= 950 for c in chunks)


def test_chunker_assigns_section_by_nearest_preceding_heading_and_carries_across_pages():
    page_a = PageText(
        page=7,
        text=(
            "Human bone marrow transplant details continue here.\n\n"
            "SCOPE OF COVER\n\n"
            "1. Room, Boarding and Nursing Expense subject to limits.\n"
        ),
    )
    page_b = PageText(
        page=8,
        text="2. Medical Practitioner fees subject to a limit of 25% of Sum Assured.\n",
    )
    chunks = chunk_pages([page_a, page_b])
    first_page_chunks = [c for c in chunks if c.page == 7]
    assert first_page_chunks[0].section == "Preamble"
    assert first_page_chunks[1].section == "Scope of Cover"
    second_page_chunks = [c for c in chunks if c.page == 8]
    assert second_page_chunks[0].section == "Scope of Cover"
