"""End-to-end retrieval test against the real built index.

Requires `python -m src.ingestion.build_index` to have been run first
(these are integration tests, not unit tests - they exercise the real
embedding model, Chroma collection, BM25 index, and cross-encoder
reranker together against the actual policy text).
"""

import pytest

from src.retrieval.retriever import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


def test_hybrid_retriever_returns_reranked_chunks_with_metadata(retriever):
    chunks, counts = retriever.retrieve("waiting period for pre-existing diseases", top_k=3)
    assert len(chunks) <= 3
    assert all(c.page > 0 and c.section for c in chunks)
    assert counts["reranked_k"] == len(chunks)


def test_hybrid_retriever_surfaces_the_known_gold_chunk_for_cosmetic_exclusion(retriever):
    chunks, _ = retriever.retrieve("is cosmetic surgery excluded under the policy?", top_k=5)
    retrieved_ids = {c.chunk_id for c in chunks}
    assert "chunk-0050" in retrieved_ids
