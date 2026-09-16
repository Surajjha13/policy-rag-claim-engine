"""Sparse lexical retrieval via BM25.

Complements dense retrieval: BM25 is strong at exact-term matches (e.g. a
reviewer's query containing the literal word "sub-limit" or "48 months")
that an embedding model can sometimes blur together with semantically
similar but factually different clauses.
"""

import pickle

from rank_bm25 import BM25Okapi

from src.config import settings


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseIndex:
    def __init__(self):
        with open(f"{settings.index_dir}/bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._bm25: BM25Okapi = data["bm25"]
        self._chunks: list[dict] = data["chunks"]

    def search(self, query: str, k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in ranked]
