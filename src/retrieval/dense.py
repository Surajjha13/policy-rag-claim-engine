"""Dense semantic retrieval over a persisted FAISS index.

Uses FAISS rather than Chroma (both are explicitly acceptable per the
assignment's tech guidance). Chroma's PyPI package pulls in a large,
mostly-unrelated dependency tree for our local-only use case - grpc, a
Kubernetes client, onnxruntime, and the full OpenTelemetry SDK, none of
which a single-process `PersistentClient` actually needs at runtime; they
exist for chromadb's optional client-server/observability modes. FAISS's
wheel is self-contained (bundles its own C++ build, only depends on numpy),
which mattered concretely in this project's very slow/unstable network
environment. The trade-off: FAISS is a pure vector index with no built-in
metadata store, so this module keeps the chunk metadata as a parallel,
index-aligned list itself (see `build_index.py`).
"""

import pickle

import faiss
from sentence_transformers import SentenceTransformer

from src.config import settings


class DenseIndex:
    def __init__(self):
        self._index = faiss.read_index(f"{settings.index_dir}/faiss.index")
        with open(f"{settings.index_dir}/faiss_chunks.pkl", "rb") as f:
            self._chunks: list[dict] = pickle.load(f)
        self._embedder = SentenceTransformer(settings.embedding_model)

    def search(self, query: str, k: int) -> list[dict]:
        vec = self._embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(vec, min(k, len(self._chunks)))
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            out.append({**chunk, "score": float(score)})
        return out
