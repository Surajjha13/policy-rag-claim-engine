"""One-time (well, one-per-policy-change) offline index build.

Run as `python -m src.ingestion.build_index`. Produces everything the
request-time retriever (`src/retrieval/`) reads: a FAISS index for dense
search (plus its index-aligned chunk-metadata list, since FAISS itself only
stores vectors), a pickled BM25 index for sparse search, and a
human-readable `chunks.json` dump (used during development to look up real
chunk_ids for `eval/gold_evidence.json` and `eval/expected_outcomes.json`).
"""

import json
import pickle
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_pages
from src.ingestion.pdf_parser import extract_pages
from src.retrieval.sparse import tokenize


def build_index() -> list[Chunk]:
    out_dir = Path(settings.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(settings.policy_pdf_path)
    chunks = chunk_pages(pages)
    (out_dir / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2), encoding="utf-8"
    )

    embedder = SentenceTransformer(settings.embedding_model)
    embeddings = embedder.encode(
        [c.text for c in chunks], show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")

    # Inner product over L2-normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # FAISS only stores vectors, not metadata - this list's order IS the
    # index: FAISS returns integer positions, and position i here must be
    # exactly the chunk whose embedding was added at position i above.
    with open(out_dir / "faiss_chunks.pkl", "wb") as f:
        pickle.dump([c.model_dump() for c in chunks], f)

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": [c.model_dump() for c in chunks]}, f)

    return chunks


if __name__ == "__main__":
    built = build_index()
    print(f"Indexed {len(built)} chunks into {settings.index_dir}")
