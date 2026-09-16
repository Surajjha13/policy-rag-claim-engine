"""HybridRetriever: the single entry point agents use for policy evidence.

Pipeline: dense top-K + sparse top-K -> RRF fusion -> cross-encoder rerank
-> top-N EvidenceChunk objects, plus a `counts` dict exposing how many
candidates survived each stage (dense_k/sparse_k/fused_k/reranked_k). The
Policy Evidence Agent attaches that dict straight into the trace so
retrieval/citation quality is measurable without exposing model reasoning.
"""

from src.config import settings
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import Reranker
from src.retrieval.sparse import SparseIndex
from src.schemas.agent_state import EvidenceChunk


class HybridRetriever:
    def __init__(self):
        self._dense = DenseIndex()
        self._sparse = SparseIndex()
        self._reranker = Reranker()

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[EvidenceChunk], dict]:
        top_k = top_k or settings.top_k_reranked
        dense_hits = self._dense.search(query, settings.top_k_dense)
        sparse_hits = self._sparse.search(query, settings.top_k_sparse)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
        reranked = self._reranker.rerank(query, fused, top_k)
        chunks = [
            EvidenceChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                page=r["page"],
                section=r["section"],
                fused_score=r.get("fused_score", 0.0),
                rerank_score=r.get("rerank_score", 0.0),
            )
            for r in reranked
        ]
        counts = {
            "dense_k": len(dense_hits),
            "sparse_k": len(sparse_hits),
            "fused_k": len(fused),
            "reranked_k": len(chunks),
        }
        return chunks, counts
