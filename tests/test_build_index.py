import pickle
from pathlib import Path

from src.config import settings
from src.ingestion.build_index import build_index


def test_build_index_creates_expected_artifacts():
    chunks = build_index()
    out_dir = Path(settings.index_dir)
    assert (out_dir / "chunks.json").exists()
    assert (out_dir / "bm25.pkl").exists()
    assert (out_dir / "faiss.index").exists()
    assert (out_dir / "faiss_chunks.pkl").exists()
    with open(out_dir / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    with open(out_dir / "faiss_chunks.pkl", "rb") as f:
        faiss_chunks = pickle.load(f)
    assert len(data["chunks"]) == len(chunks)
    assert len(faiss_chunks) == len(chunks)
    assert len(chunks) > 20
