"""Cross-encoder reranking stage.

Dense/sparse fusion is fast but coarse (bi-encoder embeddings compress a
whole chunk into one vector). A cross-encoder scores the (query, chunk)
pair jointly, which is far more accurate at judging "does this specific
clause actually answer this specific question" - at the cost of only being
affordable on the small fused candidate set, never the whole corpus.
"""

from sentence_transformers import CrossEncoder

from src.config import settings


class Reranker:
    def __init__(self):
        self._model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
