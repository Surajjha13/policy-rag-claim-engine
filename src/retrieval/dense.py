"""Dense semantic retrieval over the persisted Chroma collection."""

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings


class DenseIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=f"{settings.index_dir}/chroma")
        self._collection = self._client.get_collection("policy_chunks")
        self._embedder = SentenceTransformer(settings.embedding_model)

    def search(self, query: str, k: int) -> list[dict]:
        vec = self._embedder.encode([query]).tolist()
        res = self._collection.query(query_embeddings=vec, n_results=k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append(
                {
                    "chunk_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "page": res["metadatas"][0][i]["page"],
                    "section": res["metadatas"][0][i]["section"],
                    "score": 1.0 - res["distances"][0][i],
                }
            )
        return out
