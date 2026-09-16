# Retrieval module (`src/retrieval/`)

Everything here answers one question: given a natural-language question
from the Case Analysis Agent's checklist (e.g. "what is the waiting period
for pre-existing diseases?"), which policy chunks actually answer it, ranked
by how well they answer it.

## Why hybrid (dense + sparse), not just one

- **`dense.py` - `DenseIndex`**: wraps a persisted Chroma collection and a
  `sentence-transformers` bi-encoder (`bge-small-en-v1.5`). Good at semantic
  matches - "does the patient's condition count as a pre-existing disease"
  can match a chunk that never uses those exact words.
- **`sparse.py` - `SparseIndex`**: wraps `rank_bm25.BM25Okapi` over the same
  chunks. Good at exact-term matches - a query containing "48 months" or
  "cosmetic" should surface the chunk containing that literal term even if
  the bi-encoder's embedding space happens to rank a semantically-similar
  but factually-different clause higher.

Relying on only one would systematically miss the other's failure mode; the
assignment's rubric explicitly flags "vector search only" as a red flag.

## Why Reciprocal Rank Fusion (`fusion.py`), not weighted score blending

Dense cosine-similarity scores and BM25 scores live on incomparable scales
(dense scores cluster near 1.0; BM25 scores are unbounded and
corpus-size-dependent). Normalizing and weighting them against each other is
fragile and would need per-corpus tuning. RRF only looks at each chunk's
**rank position** in each list:

```python
score(chunk) = sum over retrievers of 1 / (k + rank_in_that_list + 1)
```

with `k=60` (the standard default from the original RRF paper - large enough
that rank differences deep in the list stop mattering much). This rewards a
chunk both retrievers agree on without needing either retriever's raw scores
to mean the same thing.

## Why rerank after fusion (`rerank.py`), not just take fusion's top-k

Dense/sparse retrieval both score a whole chunk against a query using a
single vector or bag-of-words comparison - fast, but coarse. A cross-encoder
(`bge-reranker-base`) scores the **(query, chunk) pair jointly**, letting it
notice things like "this chunk mentions '48 months' but in the context of
Break-in-Policy, not Pre-Existing-Diseases" - a distinction a bi-encoder or
BM25 comparison can't make from vectors/token-overlap alone. This is only
affordable because it runs on the small fused candidate set (≤20 chunks),
never the whole ~118-chunk corpus.

## `retriever.py` - `HybridRetriever.retrieve(query, top_k) -> (list[EvidenceChunk], dict)`

The single entry point every agent uses. Pipeline per call:

```
dense.search(query, TOP_K_DENSE)   sparse.search(query, TOP_K_SPARSE)
                \                          /
                 v                        v
              reciprocal_rank_fusion([dense_hits, sparse_hits])
                          |
                          v
                 reranker.rerank(query, fused, TOP_K_RERANKED)
                          |
                          v
              list[EvidenceChunk] + counts dict
```

The `counts` dict (`dense_k`, `sparse_k`, `fused_k`, `reranked_k`) is
returned alongside the chunks specifically so the Policy Evidence Agent can
attach it to the trace - this is the "retrieval result counts" the
assignment's trace requirement asks for, and what `eval/run_eval.py` uses to
compute recall@k against the hand-labeled gold set.

**Why `HybridRetriever` is constructed once and memoized**
(`orchestrator/pipeline.py::get_retriever`): building it loads two ML models
(the embedding model and the cross-encoder) - expensive per call, cheap to
keep warm across requests within one process.
