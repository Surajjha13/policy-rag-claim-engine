from src.config import settings
from src.ingestion.pdf_parser import extract_pages


def test_extract_pages_returns_nonempty_text_for_every_page():
    pages = extract_pages(settings.policy_pdf_path)
    assert len(pages) == 17
    assert all(p.text.strip() for p in pages)


def test_extract_pages_preserves_one_indexed_page_numbers():
    pages = extract_pages(settings.policy_pdf_path)
    assert [p.page for p in pages] == list(range(1, len(pages) + 1))


def test_first_page_contains_known_policy_heading():
    pages = extract_pages(settings.policy_pdf_path)
    assert "DEFINITIONS" in pages[0].text
