"""Reciprocal Rank Fusion (RRF): combines dense and sparse result rankings.

RRF is used instead of raw score blending because dense (cosine similarity)
and sparse (BM25) scores live on incomparable scales - normalizing and
weighting them is fragile and dataset-specific. RRF only looks at each
result's RANK in each list, so it works regardless of score scale, and it
naturally rewards a chunk that both retrievers agree is relevant.
"""


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payload.setdefault(cid, item)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**payload[cid], "fused_score": score} for cid, score in fused]
