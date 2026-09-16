"""One-time (well, one-per-policy-change) offline index build.

Run as `python -m src.ingestion.build_index`. Produces everything the
request-time retriever (`src/retrieval/`) reads: a persisted Chroma
collection for dense search, a pickled BM25 index for sparse search, and a
human-readable `chunks.json` dump (used during development to look up real
chunk_ids for `eval/gold_evidence.json` and `eval/expected_outcomes.json`).
"""

import json
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_pages
from src.ingestion.pdf_parser import extract_pages
from src.retrieval.sparse import tokenize
from rank_bm25 import BM25Okapi


def build_index() -> list[Chunk]:
    out_dir = Path(settings.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(settings.policy_pdf_path)
    chunks = chunk_pages(pages)
    (out_dir / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2), encoding="utf-8"
    )

    embedder = SentenceTransformer(settings.embedding_model)
    embeddings = embedder.encode([c.text for c in chunks], show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=str(out_dir / "chroma"))
    # Rebuilding from scratch each run keeps the index in sync with the
    # current chunker output - a stale collection from a previous chunking
    # scheme would silently mix old and new chunk_ids otherwise.
    try:
        client.delete_collection("policy_chunks")
    except Exception:
        pass
    collection = client.create_collection("policy_chunks")
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"page": c.page, "section": c.section} for c in chunks],
    )

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": [c.model_dump() for c in chunks]}, f)

    return chunks


if __name__ == "__main__":
    built = build_index()
    print(f"Indexed {len(built)} chunks into {settings.index_dir}")
