# Policy-Aware Multi-Agent RAG Claim Decision Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a FastAPI + Streamlit system that runs a 5-agent workflow over hybrid (dense+BM25+rerank) retrieval on the supplied health-insurance policy PDF to produce structured, cited, abstention-capable claim decisions, plus a reproducible evaluation harness over 12 supplied + 5+ custom cases.

**Architecture:** A one-time offline ingestion pipeline parses the policy PDF into a hand-curated section map, then clause-aware chunks with page/section metadata, embedded into Chroma (dense) and indexed with BM25 (sparse). At request time, a deterministic Python orchestrator runs five agents in sequence — Case Analysis → Policy Evidence → Coverage & Exclusion → Decision → Validation — passing typed Pydantic state between them (no free-form text handoff), with one bounded retry loop when Validation fails, and a hard fallback to `NEEDS_REVIEW` if it fails twice or required evidence/fields are missing.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Streamlit, PyMuPDF (`fitz`), `sentence-transformers` (BGE small embeddings + BGE reranker cross-encoder, CPU), `rank_bm25`, ChromaDB (local persistent), `litellm` as an LLM-provider-agnostic client (default: Groq free-tier Llama-3.3-70B, swappable via env var), `pytest`, Docker, deployed to Render (backend) + Streamlit Community Cloud (frontend).

**Spec:** `E:\assignment\Aptino_AI_Engineer_Take_Home_Assignment_FINAL.docx` (extracted text below); input materials in `E:\assignment\Aptino_Candidate_Package_FINAL\` (`candidate_data/public_test_cases.json` — 12 synthetic cases, do not modify; `policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf`; `schema/claim_case_schema.md`).

## Global Constraints

- Policy PDF is the sole authoritative source for policy conclusions — never let an agent answer from general insurance/medical knowledge; every material claim needs a citation resolvable to `(source, page, section, chunk_id)`.
- Never modify `candidate_data/public_test_cases.json`; new cases go in `eval/custom_cases.json`.
- No hidden chain-of-thought anywhere in API responses or UI — `trace[]` carries agent name, action, elapsed_ms, and counts only.
- Abstain (`NEEDS_REVIEW`) rather than guess whenever evidence/fields are insufficient; at least 2 of the custom cases must correctly resolve to `NEEDS_REVIEW`.
- Secrets only via environment variables (`.env`, never committed); ship `.env.example`.
- At least 3 genuinely specialized agents exchanging structured (Pydantic) state — not the same prompt fanned out.
- Must run locally from documented steps and be deployed to a publicly reachable frontend + backend (free/low-cost tier acceptable).
- `requirements.txt` pinned; `pytest` suite covering chunker, fusion, reranker wiring, agent I/O schemas, validation grounding logic, and API contract/error handling.

---

## File Structure

```
aptino-claim-engine/
  requirements.txt
  .env.example
  Dockerfile
  README.md
  docs/architecture_note.md
  src/
    config.py
    schemas/
      case.py
      decision.py
      agent_state.py
    ingestion/
      policy_sections.py
      pdf_parser.py
      chunker.py
      build_index.py
    retrieval/
      dense.py
      sparse.py
      fusion.py
      rerank.py
      retriever.py
    llm/
      client.py
      prompts.py
    agents/
      base.py
      case_analysis_agent.py
      policy_evidence_agent.py
      coverage_exclusion_agent.py
      decision_agent.py
      validation_agent.py
    orchestrator/
      pipeline.py
    api/
      main.py
  frontend/
    streamlit_app.py
  eval/
    custom_cases.json
    expected_outcomes.json
    gold_evidence.json
    run_eval.py
  tests/
    test_chunker.py
    test_retrieval.py
    test_agents_schema.py
    test_validation_agent.py
    test_api.py
  index_store/            # generated, gitignored except .gitkeep
```

---

### Task 1: Project scaffold, config, and schemas

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`
- Create: `src/config.py`
- Create: `src/schemas/case.py`, `src/schemas/decision.py`, `src/schemas/agent_state.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `ClaimCase` (input model), `DecisionResponse`, `Citation`, `ValidationResult`, `TraceEvent` (API contract models), `CaseState`, `InvestigationItem`, `EvidenceChunk`, `EvidenceBundle`, `DimensionFinding`, `CoverageFindings`, `DraftDecision`, `PipelineState` (inter-agent state) — every later task imports from these three files only, never redefines fields.
- Consumes: nothing (first task).

- [ ] **Step 1: Write `requirements.txt` and `.env.example`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
streamlit==1.38.0
pymupdf==1.24.10
sentence-transformers==3.1.1
rank-bm25==0.2.2
chromadb==0.5.5
litellm==1.48.0
python-dotenv==1.0.1
requests==2.32.3
pytest==8.3.3
httpx==0.27.2
```

`.env.example`:
```
LLM_PROVIDER=groq
LLM_API_KEY=your_key_here
LLM_MODEL=groq/llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-base
INDEX_DIR=./index_store
POLICY_PDF_PATH=../Aptino_Candidate_Package_FINAL/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_RERANKED=4
VALIDATION_FAIL_RETRY_LIMIT=1
```

- [ ] **Step 2: Write `src/config.py`**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_model: str = "groq/llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    index_dir: str = "./index_store"
    policy_pdf_path: str = "../Aptino_Candidate_Package_FINAL/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_reranked: int = 4
    validation_fail_retry_limit: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 3: Write `src/schemas/case.py`**

```python
from datetime import date
from pydantic import BaseModel, ConfigDict

class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int

class Hospital(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    network_provider: bool | None = None

class Treatment(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    admission_hours: float | None = None
    diagnosis: str
    procedure: str | None = None
    pre_existing: bool | None = None
    experimental: bool | None = None

class ClaimCase(BaseModel):
    model_config = ConfigDict(extra="allow")
    case_id: str
    policy_id: str
    policy_start_date: date
    claim_date: date
    sum_insured_inr: float
    continuous_coverage_months: int | None = None
    prior_insurer_continuous_years: int | None = None
    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: dict[str, float] = {}
    documents: list[str] = []
    task: str
```

- [ ] **Step 4: Write `src/schemas/decision.py`**

```python
from typing import Literal
from pydantic import BaseModel

DecisionStatus = Literal[
    "ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE", "NEEDS_REVIEW",
]

class Citation(BaseModel):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str

class ValidationResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    unsupported_claims: list[str] = []

class TraceEvent(BaseModel):
    agent: str
    action: str
    elapsed_ms: float
    detail: dict = {}

class DecisionResponse(BaseModel):
    case_id: str
    decision: DecisionStatus
    confidence: float
    key_findings: list[str]
    applicable_limits: list[str]
    missing_evidence: list[str]
    citations: list[Citation]
    validation: ValidationResult
    trace: list[TraceEvent]
```

- [ ] **Step 5: Write `src/schemas/agent_state.py`**

```python
from pydantic import BaseModel
from src.schemas.case import ClaimCase
from src.schemas.decision import Citation, ValidationResult, TraceEvent, DecisionStatus

class InvestigationItem(BaseModel):
    dimension: str
    question: str
    required: bool = True

class CaseState(BaseModel):
    case: ClaimCase
    decision_dimensions: list[str]
    missing_fields: list[str]
    checklist: list[InvestigationItem]

class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    section: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0

class EvidenceBundle(BaseModel):
    per_question: dict[str, list[EvidenceChunk]]

class DimensionFinding(BaseModel):
    dimension: str
    status: str
    explanation: str
    evidence_chunk_ids: list[str]
    confidence: float

class CoverageFindings(BaseModel):
    findings: list[DimensionFinding]

class DraftDecision(BaseModel):
    decision: DecisionStatus
    confidence: float
    key_findings: list[str]
    applicable_limits: list[str]
    missing_evidence: list[str]
    citations: list[Citation]

class PipelineState(BaseModel):
    case_state: CaseState | None = None
    evidence: EvidenceBundle | None = None
    coverage: CoverageFindings | None = None
    draft_decision: DraftDecision | None = None
    validation: ValidationResult | None = None
    trace: list[TraceEvent] = []
    retry_count: int = 0
```

- [ ] **Step 6: Write the schema test**

```python
# tests/test_schemas.py
from src.schemas.case import ClaimCase

def test_claim_case_tolerates_unknown_fields():
    raw = {
        "case_id": "X-1", "policy_id": "P", "policy_start_date": "2025-01-01",
        "claim_date": "2026-01-01", "sum_insured_inr": 500000,
        "patient": {"age": 30, "gender": "F"},
        "hospital": {"name": "H"},
        "treatment": {"type": "inpatient", "diagnosis": "flu"},
        "task": "decide",
        "some_unmodeled_field": "ignore me",
    }
    case = ClaimCase.model_validate(raw)
    assert case.case_id == "X-1"
    assert case.patient.age == 30
```

- [ ] **Step 7: Run `pytest tests/test_schemas.py -v`** — expect PASS.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example src/config.py src/schemas tests/test_schemas.py
git commit -m "feat: project scaffold, config, and API/agent-state schemas"
```

---

### Task 2: Policy section map + PDF parser

**Files:**
- Create: `src/ingestion/policy_sections.py`, `src/ingestion/pdf_parser.py`
- Test: `tests/test_pdf_parser.py`

**Interfaces:**
- Consumes: `settings.policy_pdf_path` from Task 1.
- Produces: `SECTION_MAP: list[dict]` (each `{"section": str, "start_page": int, "end_page": int}`), `extract_pages(pdf_path: str) -> list[PageText]` where `PageText = {"page": int, "text": str}`.

- [ ] **Step 1: Manually read the policy PDF's table of contents/headings once** (open it in a viewer) and hand-write `SECTION_MAP` in `src/ingestion/policy_sections.py` — e.g.:

```python
SECTION_MAP = [
    {"section": "Preamble & Definitions", "start_page": 1, "end_page": 6},
    {"section": "Scope of Cover", "start_page": 7, "end_page": 9},
    {"section": "Waiting Periods", "start_page": 10, "end_page": 11},
    {"section": "Exclusions", "start_page": 12, "end_page": 16},
    {"section": "Sub-limits & Co-payment", "start_page": 17, "end_page": 18},
    {"section": "Claims Procedure", "start_page": 19, "end_page": 21},
    {"section": "General Terms & Conditions", "start_page": 22, "end_page": 24},
]

def section_for_page(page: int) -> str:
    for entry in SECTION_MAP:
        if entry["start_page"] <= page <= entry["end_page"]:
            return entry["section"]
    return "Unclassified"
```

(Exact page numbers must be corrected against the real PDF while building this — open `Aptino_Candidate_Package_FINAL/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf` and note where each numbered section starts/ends; this hand curation is what makes chunking "meaningful" instead of naive fixed-size splitting.)

- [ ] **Step 2: Write `src/ingestion/pdf_parser.py`**

```python
import fitz  # PyMuPDF
from pydantic import BaseModel

class PageText(BaseModel):
    page: int
    text: str

def extract_pages(pdf_path: str) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages = [PageText(page=i + 1, text=doc[i].get_text("text")) for i in range(len(doc))]
    doc.close()
    return pages
```

- [ ] **Step 3: Write the test**

```python
# tests/test_pdf_parser.py
from src.config import settings
from src.ingestion.pdf_parser import extract_pages
from src.ingestion.policy_sections import section_for_page, SECTION_MAP

def test_extract_pages_returns_nonempty_text():
    pages = extract_pages(settings.policy_pdf_path)
    assert len(pages) > 10
    assert any(p.text.strip() for p in pages)

def test_section_map_covers_all_pages():
    pages = extract_pages(settings.policy_pdf_path)
    last_page = len(pages)
    assert SECTION_MAP[-1]["end_page"] >= last_page
    assert section_for_page(1) != "Unclassified"
```

- [ ] **Step 4: Run `pytest tests/test_pdf_parser.py -v`**, fix `SECTION_MAP` page ranges against the actual PDF until both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/policy_sections.py src/ingestion/pdf_parser.py tests/test_pdf_parser.py
git commit -m "feat: PDF text extraction and hand-curated policy section map"
```

---

### Task 3: Clause-aware chunker

**Files:**
- Create: `src/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `PageText` list from Task 2, `section_for_page` from Task 2.
- Produces: `Chunk` model `{"chunk_id": str, "text": str, "page": int, "section": str}` and `chunk_pages(pages: list[PageText]) -> list[Chunk]`.

- [ ] **Step 1: Write the chunker**

```python
import re
from pydantic import BaseModel
from src.ingestion.pdf_parser import PageText
from src.ingestion.policy_sections import section_for_page

class Chunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    section: str

CLAUSE_PATTERN = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,2})[\.\)]\s+")
MAX_CHUNK_CHARS = 900

def _split_clauses(text: str) -> list[str]:
    matches = list(CLAUSE_PATTERN.finditer(text))
    if not matches:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras or ([text.strip()] if text.strip() else [])
    pieces = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
    return pieces

def _split_long(piece: str) -> list[str]:
    if len(piece) <= MAX_CHUNK_CHARS:
        return [piece]
    sentences = re.split(r"(?<=[.;])\s+", piece)
    out, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 > MAX_CHUNK_CHARS and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf.strip())
    return out

def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        section = section_for_page(page.page)
        for clause in _split_clauses(page.text):
            for sub in _split_long(clause):
                if len(sub) < 20:
                    continue
                counter += 1
                chunks.append(Chunk(
                    chunk_id=f"chunk-{counter:04d}",
                    text=sub,
                    page=page.page,
                    section=section,
                ))
    return chunks
```

- [ ] **Step 2: Write the test**

```python
# tests/test_chunker.py
from src.ingestion.pdf_parser import PageText
from src.ingestion.chunker import chunk_pages

def test_chunker_splits_on_numbered_clauses_and_keeps_metadata():
    page = PageText(page=10, text=(
        "4.1 Waiting period for pre-existing diseases is 48 months.\n\n"
        "4.2 Initial waiting period is 30 days from policy inception.\n"
    ))
    chunks = chunk_pages([page])
    assert len(chunks) == 2
    assert chunks[0].page == 10
    assert "48 months" in chunks[0].text
    assert chunks[0].chunk_id != chunks[1].chunk_id

def test_chunker_never_produces_chunk_over_max_chars():
    long_text = "5.1 " + ("Sub-limit clause text. " * 200)
    page = PageText(page=17, text=long_text)
    chunks = chunk_pages([page])
    assert all(len(c.text) <= 950 for c in chunks)
```

- [ ] **Step 3: Run `pytest tests/test_chunker.py -v`** — expect PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: clause-aware policy chunker with page/section metadata"
```

---

### Task 4: Build the index (dense + sparse) — CLI script

**Files:**
- Create: `src/ingestion/build_index.py`
- Test: `tests/test_build_index.py`

**Interfaces:**
- Consumes: `extract_pages`, `chunk_pages`, `settings`.
- Produces: on-disk artifacts at `settings.index_dir`: a persisted Chroma collection named `policy_chunks`, and `bm25.pkl` (pickled `{"bm25": BM25Okapi, "chunks": list[Chunk]}`), plus `chunks.json` (all chunks, human-inspectable).

- [ ] **Step 1: Write `src/ingestion/build_index.py`**

```python
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages, Chunk

def tokenize(text: str) -> list[str]:
    return text.lower().split()

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
    collection = client.get_or_create_collection("policy_chunks")
    collection.upsert(
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
```

- [ ] **Step 2: Write the test**

```python
# tests/test_build_index.py
import pickle
from pathlib import Path
from src.config import settings
from src.ingestion.build_index import build_index

def test_build_index_creates_expected_artifacts():
    chunks = build_index()
    out_dir = Path(settings.index_dir)
    assert (out_dir / "chunks.json").exists()
    assert (out_dir / "bm25.pkl").exists()
    with open(out_dir / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    assert len(data["chunks"]) == len(chunks)
    assert len(chunks) > 20
```

- [ ] **Step 3: Run `python -m src.ingestion.build_index`**, confirm it prints a chunk count > 20, then run `pytest tests/test_build_index.py -v`.

- [ ] **Step 4: Add `index_store/` to `.gitignore`** except keep the script re-runnable at deploy/build time.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/build_index.py tests/test_build_index.py .gitignore
git commit -m "feat: index build script producing Chroma dense index + BM25 sparse index"
```

---

### Task 5: Hybrid retriever — dense, sparse, RRF fusion, cross-encoder rerank

**Files:**
- Create: `src/retrieval/dense.py`, `src/retrieval/sparse.py`, `src/retrieval/fusion.py`, `src/retrieval/rerank.py`, `src/retrieval/retriever.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: index artifacts from Task 4, `EvidenceChunk` from Task 1.
- Produces: `HybridRetriever.retrieve(query: str, top_k: int = settings.top_k_reranked) -> list[EvidenceChunk]`.

- [ ] **Step 1: `src/retrieval/dense.py`**

```python
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
            out.append({
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "page": res["metadatas"][0][i]["page"],
                "section": res["metadatas"][0][i]["section"],
                "score": 1.0 - res["distances"][0][i],
            })
        return out
```

- [ ] **Step 2: `src/retrieval/sparse.py`**

```python
import pickle
from src.config import settings
from src.ingestion.build_index import tokenize

class SparseIndex:
    def __init__(self):
        with open(f"{settings.index_dir}/bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]

    def search(self, query: str, k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {**self._chunks[i], "score": float(scores[i])}
            for i in ranked
        ]
```

- [ ] **Step 3: `src/retrieval/fusion.py`**

```python
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
```

- [ ] **Step 4: `src/retrieval/rerank.py`**

```python
from sentence_transformers import CrossEncoder
from src.config import settings

class Reranker:
    def __init__(self):
        self._model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
```

- [ ] **Step 5: `src/retrieval/retriever.py`**

```python
from src.config import settings
from src.schemas.agent_state import EvidenceChunk
from src.retrieval.dense import DenseIndex
from src.retrieval.sparse import SparseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import Reranker

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
                chunk_id=r["chunk_id"], text=r["text"], page=r["page"], section=r["section"],
                fused_score=r.get("fused_score", 0.0), rerank_score=r.get("rerank_score", 0.0),
            ) for r in reranked
        ]
        counts = {"dense_k": len(dense_hits), "sparse_k": len(sparse_hits),
                  "fused_k": len(fused), "reranked_k": len(chunks)}
        return chunks, counts
```

- [ ] **Step 6: Write the test**

```python
# tests/test_retrieval.py
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.retriever import HybridRetriever

def test_rrf_prefers_items_ranked_highly_in_both_lists():
    dense = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}]
    sparse = [{"chunk_id": "b", "text": "b"}, {"chunk_id": "a", "text": "a"}]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0]["fused_score"] >= fused[1]["fused_score"]

def test_hybrid_retriever_returns_reranked_chunks_with_metadata():
    retriever = HybridRetriever()
    chunks, counts = retriever.retrieve("waiting period for pre-existing disease", top_k=3)
    assert len(chunks) <= 3
    assert all(c.page > 0 and c.section for c in chunks)
    assert counts["reranked_k"] == len(chunks)
```

(This test requires the index built in Task 4 to exist — run `python -m src.ingestion.build_index` first if not already run in this environment.)

- [ ] **Step 7: Run `pytest tests/test_retrieval.py -v`** — expect PASS.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval tests/test_retrieval.py
git commit -m "feat: hybrid retrieval (dense + BM25 + RRF fusion + cross-encoder rerank)"
```

---

### Task 6: LLM client abstraction and prompts

**Files:**
- Create: `src/llm/client.py`, `src/llm/prompts.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `settings.llm_model`, `settings.llm_api_key`.
- Produces: `chat_json(system: str, user: str, schema_hint: str) -> dict` — calls the LLM via `litellm.completion`, asks for strict JSON, parses and returns a `dict`; raises `LLMOutputError` on unparseable output so agents can catch it and fall back to `NEEDS_REVIEW`.

- [ ] **Step 1: `src/llm/client.py`**

```python
import json
import litellm
from src.config import settings

class LLMOutputError(Exception):
    pass

def chat_json(system: str, user: str) -> dict:
    response = litellm.completion(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        messages=[
            {"role": "system", "content": system + "\nRespond with ONLY valid JSON. No prose, no markdown fences."},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        timeout=30,
    )
    content = response["choices"][0]["message"]["content"].strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"Model did not return valid JSON: {content[:200]}") from e
```

- [ ] **Step 2: `src/llm/prompts.py`** — one constant per agent (kept short; each instructs "state only your final structured answer, do not show reasoning steps"):

```python
CASE_ANALYSIS_SYSTEM = (
    "You are a claims case-analysis specialist for a health insurance policy engine. "
    "Given a structured claim case, identify which policy decision dimensions apply "
    "(choose from: waiting_period, pre_existing_disease, coverage_scope, exclusions, "
    "sub_limits, documentation_sufficiency, hospital_definition, day_care_procedure) "
    "and produce natural-language retrieval questions for each relevant dimension. "
    "Do not explain your reasoning; output only the final JSON."
)

COVERAGE_EXCLUSION_SYSTEM = (
    "You are a coverage-and-exclusions specialist. Given case facts and retrieved policy "
    "evidence, decide the status of each decision dimension using ONLY the provided evidence "
    "text. If the evidence does not clearly settle a dimension, set status to 'UNCLEAR' and "
    "confidence below 0.5. Cite evidence by chunk_id. Do not explain your reasoning; output only the final JSON."
)

DECISION_SYSTEM = (
    "You are the final decision specialist for health insurance claims. Combine the coverage "
    "findings into one decision from: ADMISSIBLE, ADMISSIBLE_WITH_LIMITS, PARTIALLY_ADMISSIBLE, "
    "NOT_ADMISSIBLE, NEEDS_REVIEW. Use NEEDS_REVIEW whenever a required dimension is UNCLEAR or "
    "evidence is missing. Every entry in key_findings/applicable_limits must map to a citation "
    "with an existing chunk_id from the findings you were given. Never invent a chunk_id, page, "
    "or number not present in the findings. Do not explain your reasoning; output only the final JSON."
)

VALIDATION_SYSTEM = (
    "You are a strict fact-checker. Given a claim statement and the exact policy chunk text it "
    "cites, answer whether the chunk text actually supports the statement. Be conservative: if "
    "unsure, answer false. Output only the final JSON: {\"supported\": true|false}."
)
```

- [ ] **Step 3: Write the test (mocked, no real API call)**

```python
# tests/test_llm_client.py
from unittest.mock import patch, MagicMock
from src.llm.client import chat_json, LLMOutputError

def test_chat_json_parses_valid_json():
    fake_response = {"choices": [{"message": {"content": '{"a": 1}'}}]}
    with patch("src.llm.client.litellm.completion", return_value=fake_response):
        assert chat_json("sys", "user") == {"a": 1}

def test_chat_json_strips_markdown_fences():
    fake_response = {"choices": [{"message": {"content": '```json\n{"a": 2}\n```'}}]}
    with patch("src.llm.client.litellm.completion", return_value=fake_response):
        assert chat_json("sys", "user") == {"a": 2}

def test_chat_json_raises_on_garbage():
    fake_response = {"choices": [{"message": {"content": "not json at all"}}]}
    with patch("src.llm.client.litellm.completion", return_value=fake_response):
        try:
            chat_json("sys", "user")
            assert False, "expected LLMOutputError"
        except LLMOutputError:
            pass
```

- [ ] **Step 4: Run `pytest tests/test_llm_client.py -v`** — expect PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm tests/test_llm_client.py
git commit -m "feat: provider-agnostic LLM JSON client and agent prompts"
```

---

### Task 7: Case Analysis Agent (mostly deterministic, LLM only for dimension/question extraction)

**Files:**
- Create: `src/agents/base.py`, `src/agents/case_analysis_agent.py`
- Test: `tests/test_case_analysis_agent.py`

**Interfaces:**
- Consumes: `ClaimCase`, `chat_json`, `CASE_ANALYSIS_SYSTEM`, `TraceEvent`.
- Produces: `run_case_analysis(case: ClaimCase, trace: list[TraceEvent]) -> CaseState`.

- [ ] **Step 1: `src/agents/base.py`**

```python
import time
from src.schemas.decision import TraceEvent

def timed(agent_name: str, action: str, trace: list[TraceEvent], fn, *args, detail: dict | None = None, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    trace.append(TraceEvent(agent=agent_name, action=action, elapsed_ms=round(elapsed_ms, 1), detail=detail or {}))
    return result
```

- [ ] **Step 2: `src/agents/case_analysis_agent.py`**

```python
import json
from src.schemas.case import ClaimCase
from src.schemas.agent_state import CaseState, InvestigationItem
from src.schemas.decision import TraceEvent
from src.llm.client import chat_json, LLMOutputError
from src.llm.prompts import CASE_ANALYSIS_SYSTEM
from src.agents.base import timed

REQUIRED_DOCS = {"claim_form", "discharge_summary", "itemized_bill"}

def _detect_missing_fields(case: ClaimCase) -> list[str]:
    missing = []
    if case.treatment.admission_hours is None:
        missing.append("treatment.admission_hours")
    if not (REQUIRED_DOCS & set(case.documents)):
        missing.append("documents (no claim_form/discharge_summary/itemized_bill present)")
    if case.treatment.pre_existing is None:
        missing.append("treatment.pre_existing")
    return missing

def _llm_extract(case: ClaimCase) -> dict:
    user = json.dumps({
        "case": case.model_dump(mode="json"),
        "instructions": "Return JSON: {\"decision_dimensions\": [...], "
                        "\"checklist\": [{\"dimension\": str, \"question\": str, \"required\": bool}]}",
    })
    return chat_json(CASE_ANALYSIS_SYSTEM, user)

def run_case_analysis(case: ClaimCase, trace: list[TraceEvent]) -> CaseState:
    missing_fields = _detect_missing_fields(case)

    def _call():
        try:
            return _llm_extract(case)
        except LLMOutputError:
            return {
                "decision_dimensions": ["coverage_scope"],
                "checklist": [{"dimension": "coverage_scope",
                               "question": f"Is {case.treatment.diagnosis} covered under the policy?",
                               "required": True}],
            }

    result = timed("CaseAnalysisAgent", "extract_dimensions_and_checklist", trace, _call)
    checklist = [InvestigationItem(**item) for item in result["checklist"]]
    return CaseState(
        case=case,
        decision_dimensions=result["decision_dimensions"],
        missing_fields=missing_fields,
        checklist=checklist,
    )
```

- [ ] **Step 3: Write the test (mocked LLM)**

```python
# tests/test_case_analysis_agent.py
from unittest.mock import patch
from src.schemas.case import ClaimCase
from src.agents.case_analysis_agent import run_case_analysis

CASE = ClaimCase.model_validate({
    "case_id": "T-1", "policy_id": "P", "policy_start_date": "2025-01-01",
    "claim_date": "2026-01-01", "sum_insured_inr": 500000,
    "patient": {"age": 30}, "hospital": {"name": "H"},
    "treatment": {"type": "inpatient", "diagnosis": "appendicitis", "admission_hours": 96},
    "documents": ["claim_form", "discharge_summary"],
    "task": "decide",
})

def test_case_analysis_flags_missing_pre_existing_field():
    trace = []
    with patch("src.agents.case_analysis_agent._llm_extract", return_value={
        "decision_dimensions": ["coverage_scope"],
        "checklist": [{"dimension": "coverage_scope", "question": "is it covered?", "required": True}],
    }):
        state = run_case_analysis(CASE, trace)
    assert "treatment.pre_existing" in state.missing_fields
    assert len(trace) == 1
    assert trace[0].agent == "CaseAnalysisAgent"

def test_case_analysis_falls_back_on_llm_error():
    from src.llm.client import LLMOutputError
    trace = []
    with patch("src.agents.case_analysis_agent._llm_extract", side_effect=LLMOutputError("bad")):
        state = run_case_analysis(CASE, trace)
    assert state.decision_dimensions == ["coverage_scope"]
```

- [ ] **Step 4: Run `pytest tests/test_case_analysis_agent.py -v`** — expect PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py src/agents/case_analysis_agent.py tests/test_case_analysis_agent.py
git commit -m "feat: Case Analysis Agent with deterministic field checks + LLM dimension extraction"
```

---

### Task 8: Policy Evidence Agent

**Files:**
- Create: `src/agents/policy_evidence_agent.py`
- Test: `tests/test_policy_evidence_agent.py`

**Interfaces:**
- Consumes: `CaseState.checklist`, `HybridRetriever`.
- Produces: `run_policy_evidence(case_state: CaseState, retriever: HybridRetriever, trace: list[TraceEvent]) -> EvidenceBundle`.

- [ ] **Step 1: Implement**

```python
from src.schemas.agent_state import CaseState, EvidenceBundle
from src.schemas.decision import TraceEvent
from src.retrieval.retriever import HybridRetriever
from src.agents.base import timed

def run_policy_evidence(case_state: CaseState, retriever: HybridRetriever, trace: list[TraceEvent]) -> EvidenceBundle:
    per_question = {}
    for item in case_state.checklist:
        chunks, counts = timed(
            "PolicyEvidenceAgent", f"retrieve:{item.dimension}", trace,
            retriever.retrieve, item.question, detail={},
        )
        trace[-1].detail = counts
        per_question[item.question] = chunks
    return EvidenceBundle(per_question=per_question)
```

- [ ] **Step 2: Write the test (fake retriever, no real index needed)**

```python
# tests/test_policy_evidence_agent.py
from src.schemas.agent_state import CaseState, InvestigationItem, EvidenceChunk
from src.schemas.case import ClaimCase
from src.agents.policy_evidence_agent import run_policy_evidence

class FakeRetriever:
    def retrieve(self, query, top_k=None):
        chunk = EvidenceChunk(chunk_id="c1", text="waiting period is 30 days", page=10, section="Waiting Periods")
        return [chunk], {"dense_k": 5, "sparse_k": 5, "fused_k": 5, "reranked_k": 1}

CASE = ClaimCase.model_validate({
    "case_id": "T-1", "policy_id": "P", "policy_start_date": "2025-01-01",
    "claim_date": "2026-01-01", "sum_insured_inr": 500000,
    "patient": {"age": 30}, "hospital": {"name": "H"},
    "treatment": {"type": "inpatient", "diagnosis": "flu"}, "task": "decide",
})

def test_policy_evidence_agent_retrieves_per_checklist_item():
    case_state = CaseState(
        case=CASE, decision_dimensions=["waiting_period"], missing_fields=[],
        checklist=[InvestigationItem(dimension="waiting_period", question="what is the waiting period?")],
    )
    trace = []
    bundle = run_policy_evidence(case_state, FakeRetriever(), trace)
    assert "what is the waiting period?" in bundle.per_question
    assert bundle.per_question["what is the waiting period?"][0].chunk_id == "c1"
    assert trace[0].detail["reranked_k"] == 1
```

- [ ] **Step 3: Run `pytest tests/test_policy_evidence_agent.py -v`** — expect PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agents/policy_evidence_agent.py tests/test_policy_evidence_agent.py
git commit -m "feat: Policy Evidence Agent wiring checklist questions to hybrid retrieval"
```

---

### Task 9: Coverage & Exclusion Agent

**Files:**
- Create: `src/agents/coverage_exclusion_agent.py`
- Test: `tests/test_coverage_exclusion_agent.py`

**Interfaces:**
- Consumes: `CaseState`, `EvidenceBundle`, `chat_json`, `COVERAGE_EXCLUSION_SYSTEM`.
- Produces: `run_coverage_exclusion(case_state: CaseState, evidence: EvidenceBundle, trace: list[TraceEvent]) -> CoverageFindings`.

- [ ] **Step 1: Implement**

```python
import json
from src.schemas.agent_state import CaseState, EvidenceBundle, CoverageFindings, DimensionFinding
from src.schemas.decision import TraceEvent
from src.llm.client import chat_json, LLMOutputError
from src.llm.prompts import COVERAGE_EXCLUSION_SYSTEM
from src.agents.base import timed

def _build_prompt_payload(case_state: CaseState, evidence: EvidenceBundle) -> dict:
    return {
        "case_facts": case_state.case.model_dump(mode="json"),
        "missing_fields": case_state.missing_fields,
        "evidence_by_question": {
            q: [{"chunk_id": c.chunk_id, "text": c.text, "page": c.page, "section": c.section} for c in chunks]
            for q, chunks in evidence.per_question.items()
        },
        "instructions": (
            "Return JSON: {\"findings\": [{\"dimension\": str, \"status\": str, "
            "\"explanation\": str, \"evidence_chunk_ids\": [str], \"confidence\": float}]}"
        ),
    }

def run_coverage_exclusion(case_state: CaseState, evidence: EvidenceBundle, trace: list[TraceEvent]) -> CoverageFindings:
    def _call():
        try:
            payload = _build_prompt_payload(case_state, evidence)
            return chat_json(COVERAGE_EXCLUSION_SYSTEM, json.dumps(payload))
        except LLMOutputError:
            return {"findings": [{
                "dimension": "coverage_scope", "status": "UNCLEAR",
                "explanation": "Model output could not be parsed.",
                "evidence_chunk_ids": [], "confidence": 0.0,
            }]}

    result = timed("CoverageExclusionAgent", "assess_dimensions", trace, _call)
    return CoverageFindings(findings=[DimensionFinding(**f) for f in result["findings"]])
```

- [ ] **Step 2: Write the test**

```python
# tests/test_coverage_exclusion_agent.py
from unittest.mock import patch
from src.schemas.agent_state import CaseState, InvestigationItem, EvidenceBundle
from src.schemas.case import ClaimCase
from src.agents.coverage_exclusion_agent import run_coverage_exclusion

CASE = ClaimCase.model_validate({
    "case_id": "T-1", "policy_id": "P", "policy_start_date": "2025-01-01",
    "claim_date": "2026-01-01", "sum_insured_inr": 500000,
    "patient": {"age": 30}, "hospital": {"name": "H"},
    "treatment": {"type": "inpatient", "diagnosis": "flu"}, "task": "decide",
})
CASE_STATE = CaseState(case=CASE, decision_dimensions=["waiting_period"], missing_fields=[],
                        checklist=[InvestigationItem(dimension="waiting_period", question="q1")])
EVIDENCE = EvidenceBundle(per_question={"q1": []})

def test_coverage_exclusion_returns_structured_findings():
    trace = []
    with patch("src.agents.coverage_exclusion_agent.chat_json", return_value={
        "findings": [{"dimension": "waiting_period", "status": "WITHIN_WAITING_PERIOD",
                       "explanation": "policy started 5 days ago", "evidence_chunk_ids": ["c1"],
                       "confidence": 0.9}],
    }):
        findings = run_coverage_exclusion(CASE_STATE, EVIDENCE, trace)
    assert findings.findings[0].status == "WITHIN_WAITING_PERIOD"
    assert trace[0].agent == "CoverageExclusionAgent"

def test_coverage_exclusion_falls_back_to_unclear_on_llm_error():
    from src.llm.client import LLMOutputError
    trace = []
    with patch("src.agents.coverage_exclusion_agent.chat_json", side_effect=LLMOutputError("x")):
        findings = run_coverage_exclusion(CASE_STATE, EVIDENCE, trace)
    assert findings.findings[0].status == "UNCLEAR"
    assert findings.findings[0].confidence == 0.0
```

- [ ] **Step 3: Run `pytest tests/test_coverage_exclusion_agent.py -v`** — expect PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agents/coverage_exclusion_agent.py tests/test_coverage_exclusion_agent.py
git commit -m "feat: Coverage & Exclusion Agent producing structured per-dimension findings"
```

---

### Task 10: Decision Agent (with abstention rules)

**Files:**
- Create: `src/agents/decision_agent.py`
- Test: `tests/test_decision_agent.py`

**Interfaces:**
- Consumes: `CaseState`, `CoverageFindings`, `chat_json`, `DECISION_SYSTEM`.
- Produces: `run_decision(case_state: CaseState, coverage: CoverageFindings, trace: list[TraceEvent], feedback: list[str] | None = None) -> DraftDecision`.

- [ ] **Step 1: Implement**

```python
import json
from src.schemas.agent_state import CaseState, CoverageFindings, DraftDecision
from src.schemas.decision import TraceEvent
from src.llm.client import chat_json, LLMOutputError
from src.llm.prompts import DECISION_SYSTEM
from src.agents.base import timed

CONFIDENCE_FLOOR = 0.55

def _forced_needs_review(case_state: CaseState, coverage: CoverageFindings) -> DraftDecision | None:
    unclear = [f for f in coverage.findings if f.status == "UNCLEAR" or f.confidence < CONFIDENCE_FLOOR]
    if case_state.missing_fields or unclear:
        reasons = list(case_state.missing_fields) + [f"{f.dimension}: {f.explanation}" for f in unclear]
        return DraftDecision(
            decision="NEEDS_REVIEW", confidence=0.4,
            key_findings=[f.explanation for f in coverage.findings if f.status != "UNCLEAR"],
            applicable_limits=[], missing_evidence=reasons, citations=[],
        )
    return None

def run_decision(case_state: CaseState, coverage: CoverageFindings, trace: list[TraceEvent],
                  feedback: list[str] | None = None) -> DraftDecision:
    forced = _forced_needs_review(case_state, coverage)
    if forced is not None:
        trace.append(TraceEvent(agent="DecisionAgent", action="abstain_missing_evidence", elapsed_ms=0.0,
                                  detail={"reasons": len(forced.missing_evidence)}))
        return forced

    payload = {
        "case_facts": case_state.case.model_dump(mode="json"),
        "findings": [f.model_dump() for f in coverage.findings],
        "revision_feedback": feedback or [],
        "instructions": (
            "Return JSON: {\"decision\": str, \"confidence\": float, \"key_findings\": [str], "
            "\"applicable_limits\": [str], \"missing_evidence\": [str], "
            "\"citations\": [{\"claim\": str, \"source\": \"policy.pdf\", \"page\": int, "
            "\"section\": str, \"chunk_id\": str}]}"
        ),
    }

    def _call():
        try:
            return chat_json(DECISION_SYSTEM, json.dumps(payload))
        except LLMOutputError:
            return {"decision": "NEEDS_REVIEW", "confidence": 0.3, "key_findings": [],
                     "applicable_limits": [], "missing_evidence": ["Decision model output was unparseable."],
                     "citations": []}

    result = timed("DecisionAgent", "combine_findings", trace, _call)
    return DraftDecision(**result)
```

- [ ] **Step 2: Write the test**

```python
# tests/test_decision_agent.py
from unittest.mock import patch
from src.schemas.agent_state import CaseState, InvestigationItem, CoverageFindings, DimensionFinding
from src.schemas.case import ClaimCase
from src.agents.decision_agent import run_decision

CASE = ClaimCase.model_validate({
    "case_id": "T-1", "policy_id": "P", "policy_start_date": "2025-01-01",
    "claim_date": "2026-01-01", "sum_insured_inr": 500000,
    "patient": {"age": 30}, "hospital": {"name": "H"},
    "treatment": {"type": "inpatient", "diagnosis": "flu"}, "task": "decide",
})

def test_decision_agent_forces_needs_review_when_missing_fields_present():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=["documents missing"], checklist=[])
    coverage = CoverageFindings(findings=[])
    trace = []
    decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "NEEDS_REVIEW"
    assert trace[0].action == "abstain_missing_evidence"

def test_decision_agent_forces_needs_review_when_finding_unclear():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    coverage = CoverageFindings(findings=[DimensionFinding(
        dimension="coverage_scope", status="UNCLEAR", explanation="no clause found",
        evidence_chunk_ids=[], confidence=0.2)])
    trace = []
    decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "NEEDS_REVIEW"

def test_decision_agent_uses_llm_when_findings_are_confident():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    coverage = CoverageFindings(findings=[DimensionFinding(
        dimension="coverage_scope", status="COVERED", explanation="matches scope of cover",
        evidence_chunk_ids=["c1"], confidence=0.9)])
    trace = []
    with patch("src.agents.decision_agent.chat_json", return_value={
        "decision": "ADMISSIBLE", "confidence": 0.9, "key_findings": ["covered"],
        "applicable_limits": [], "missing_evidence": [],
        "citations": [{"claim": "covered", "source": "policy.pdf", "page": 7, "section": "Scope of Cover", "chunk_id": "c1"}],
    }):
        decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "ADMISSIBLE"
    assert decision.citations[0].chunk_id == "c1"
```

- [ ] **Step 3: Run `pytest tests/test_decision_agent.py -v`** — expect PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agents/decision_agent.py tests/test_decision_agent.py
git commit -m "feat: Decision Agent with deterministic abstention rules for missing/unclear evidence"
```

---

### Task 11: Validation Agent (grounding check) + orchestrator pipeline with retry loop

**Files:**
- Create: `src/agents/validation_agent.py`, `src/orchestrator/pipeline.py`
- Test: `tests/test_validation_agent.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `DraftDecision`, `EvidenceBundle`, `chat_json`, `VALIDATION_SYSTEM`.
- Produces: `run_validation(decision: DraftDecision, evidence: EvidenceBundle, trace: list[TraceEvent]) -> ValidationResult`; `run_pipeline(case: ClaimCase) -> DecisionResponse`.

- [ ] **Step 1: `src/agents/validation_agent.py`**

```python
import re
from src.schemas.agent_state import DraftDecision, EvidenceBundle
from src.schemas.decision import TraceEvent, ValidationResult
from src.llm.client import chat_json, LLMOutputError
from src.llm.prompts import VALIDATION_SYSTEM
from src.agents.base import timed

def _all_chunks_by_id(evidence: EvidenceBundle) -> dict[str, str]:
    out = {}
    for chunks in evidence.per_question.values():
        for c in chunks:
            out[c.chunk_id] = c.text
    return out

def _keyword_overlap_ok(claim: str, chunk_text: str) -> bool:
    claim_tokens = {t for t in re.findall(r"[a-zA-Z0-9%]+", claim.lower()) if len(t) > 3}
    chunk_tokens = {t for t in re.findall(r"[a-zA-Z0-9%]+", chunk_text.lower())}
    if not claim_tokens:
        return True
    overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
    return overlap >= 0.3

def run_validation(decision: DraftDecision, evidence: EvidenceBundle, trace: list[TraceEvent]) -> ValidationResult:
    chunk_lookup = _all_chunks_by_id(evidence)
    unsupported = []

    def _check_all():
        for citation in decision.citations:
            chunk_text = chunk_lookup.get(citation.chunk_id)
            if chunk_text is None:
                unsupported.append(f"citation references unknown chunk_id {citation.chunk_id}")
                continue
            if not _keyword_overlap_ok(citation.claim, chunk_text):
                unsupported.append(f"low keyword overlap for claim: {citation.claim}")
                continue
            try:
                verdict = chat_json(VALIDATION_SYSTEM, f"Claim: {citation.claim}\nPolicy text: {chunk_text}")
                if not verdict.get("supported", False):
                    unsupported.append(f"model rejected support for claim: {citation.claim}")
            except LLMOutputError:
                unsupported.append(f"validation model failed to judge claim: {citation.claim}")
        return unsupported

    timed("ValidationAgent", "check_citation_grounding", trace, _check_all,
          detail={"citations_checked": len(decision.citations)})
    status = "FAIL" if unsupported else "PASS"
    return ValidationResult(status=status, unsupported_claims=unsupported)
```

- [ ] **Step 2: `src/orchestrator/pipeline.py`**

```python
from src.config import settings
from src.schemas.case import ClaimCase
from src.schemas.agent_state import PipelineState
from src.schemas.decision import DecisionResponse
from src.retrieval.retriever import HybridRetriever
from src.agents.case_analysis_agent import run_case_analysis
from src.agents.policy_evidence_agent import run_policy_evidence
from src.agents.coverage_exclusion_agent import run_coverage_exclusion
from src.agents.decision_agent import run_decision
from src.agents.validation_agent import run_validation

_retriever: HybridRetriever | None = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def run_pipeline(case: ClaimCase) -> DecisionResponse:
    state = PipelineState()
    state.case_state = run_case_analysis(case, state.trace)
    state.evidence = run_policy_evidence(state.case_state, get_retriever(), state.trace)
    state.coverage = run_coverage_exclusion(state.case_state, state.evidence, state.trace)
    state.draft_decision = run_decision(state.case_state, state.coverage, state.trace)
    state.validation = run_validation(state.draft_decision, state.evidence, state.trace)

    while state.validation.status == "FAIL" and state.retry_count < settings.validation_fail_retry_limit:
        state.retry_count += 1
        state.draft_decision = run_decision(
            state.case_state, state.coverage, state.trace,
            feedback=state.validation.unsupported_claims,
        )
        state.validation = run_validation(state.draft_decision, state.evidence, state.trace)

    if state.validation.status == "FAIL":
        state.draft_decision.decision = "NEEDS_REVIEW"
        state.draft_decision.missing_evidence.append(
            "Validation could not confirm evidence support for the decision; escalated for manual review."
        )

    return DecisionResponse(
        case_id=case.case_id,
        decision=state.draft_decision.decision,
        confidence=state.draft_decision.confidence,
        key_findings=state.draft_decision.key_findings,
        applicable_limits=state.draft_decision.applicable_limits,
        missing_evidence=state.draft_decision.missing_evidence,
        citations=state.draft_decision.citations,
        validation=state.validation,
        trace=state.trace,
    )
```

- [ ] **Step 3: Write `tests/test_validation_agent.py`**

```python
from unittest.mock import patch
from src.schemas.agent_state import DraftDecision, EvidenceBundle, EvidenceChunk
from src.schemas.decision import Citation
from src.agents.validation_agent import run_validation

EVIDENCE = EvidenceBundle(per_question={"q1": [
    EvidenceChunk(chunk_id="c1", text="The waiting period for pre-existing diseases is 48 months.",
                  page=10, section="Waiting Periods"),
]})

def test_validation_fails_on_unknown_chunk_id():
    decision = DraftDecision(decision="ADMISSIBLE", confidence=0.9, key_findings=[],
                              applicable_limits=[], missing_evidence=[],
                              citations=[Citation(claim="x", source="policy.pdf", page=1, section="s", chunk_id="does-not-exist")])
    trace = []
    result = run_validation(decision, EVIDENCE, trace)
    assert result.status == "FAIL"

def test_validation_passes_when_claim_matches_chunk_and_model_agrees():
    decision = DraftDecision(decision="ADMISSIBLE", confidence=0.9, key_findings=[],
                              applicable_limits=[], missing_evidence=[],
                              citations=[Citation(claim="waiting period for pre-existing diseases is 48 months",
                                                    source="policy.pdf", page=10, section="Waiting Periods", chunk_id="c1")])
    trace = []
    with patch("src.agents.validation_agent.chat_json", return_value={"supported": True}):
        result = run_validation(decision, EVIDENCE, trace)
    assert result.status == "PASS"
```

- [ ] **Step 4: Write `tests/test_pipeline.py`** (fully mocked agent functions to test wiring/retry logic in isolation)

```python
from unittest.mock import patch
from src.schemas.case import ClaimCase
from src.schemas.agent_state import CaseState, EvidenceBundle, CoverageFindings, DraftDecision
from src.schemas.decision import ValidationResult
from src.orchestrator import pipeline

CASE = ClaimCase.model_validate({
    "case_id": "T-1", "policy_id": "P", "policy_start_date": "2025-01-01",
    "claim_date": "2026-01-01", "sum_insured_inr": 500000,
    "patient": {"age": 30}, "hospital": {"name": "H"},
    "treatment": {"type": "inpatient", "diagnosis": "flu"}, "task": "decide",
})

def test_pipeline_escalates_to_needs_review_after_exhausting_retry():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    draft = DraftDecision(decision="ADMISSIBLE", confidence=0.8, key_findings=["x"],
                            applicable_limits=[], missing_evidence=[], citations=[])
    fail = ValidationResult(status="FAIL", unsupported_claims=["x"])
    with patch("src.orchestrator.pipeline.run_case_analysis", return_value=case_state), \
         patch("src.orchestrator.pipeline.get_retriever", return_value=None), \
         patch("src.orchestrator.pipeline.run_policy_evidence", return_value=EvidenceBundle(per_question={})), \
         patch("src.orchestrator.pipeline.run_coverage_exclusion", return_value=CoverageFindings(findings=[])), \
         patch("src.orchestrator.pipeline.run_decision", return_value=draft), \
         patch("src.orchestrator.pipeline.run_validation", return_value=fail):
        response = pipeline.run_pipeline(CASE)
    assert response.decision == "NEEDS_REVIEW"
    assert response.validation.status == "FAIL"
    assert "escalated for manual review" in " ".join(response.missing_evidence)
```

- [ ] **Step 5: Run `pytest tests/test_validation_agent.py tests/test_pipeline.py -v`** — expect PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/validation_agent.py src/orchestrator tests/test_validation_agent.py tests/test_pipeline.py
git commit -m "feat: Validation Agent grounding checks and full 5-agent pipeline with bounded retry loop"
```

---

### Task 12: FastAPI backend

**Files:**
- Create: `src/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `run_pipeline`, `ClaimCase`, `DecisionResponse`.
- Produces: `POST /analyze`, `GET /health` on a FastAPI app importable as `src.api.main:app`.

- [ ] **Step 1: Implement**

```python
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.schemas.case import ClaimCase
from src.schemas.decision import DecisionResponse
from src.orchestrator.pipeline import run_pipeline, get_retriever

app = FastAPI(title="Aptino Policy-Aware Claim Decision Engine")

@app.get("/health")
def health():
    try:
        get_retriever()
        index_ready = True
    except Exception:
        index_ready = False
    return {"status": "ok" if index_ready else "degraded", "index_ready": index_ready}

@app.post("/analyze", response_model=DecisionResponse)
def analyze(payload: dict):
    try:
        case = ClaimCase.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    try:
        return run_pipeline(case)
    except Exception as e:
        return DecisionResponse(
            case_id=payload.get("case_id", "UNKNOWN"),
            decision="NEEDS_REVIEW", confidence=0.0,
            key_findings=[], applicable_limits=[],
            missing_evidence=[f"Internal pipeline error: {type(e).__name__}"],
            citations=[], validation={"status": "FAIL", "unsupported_claims": []},
            trace=[],
        )
```

- [ ] **Step 2: Write the test**

```python
# tests/test_api.py
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.schemas.decision import DecisionResponse, ValidationResult

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()

def test_analyze_rejects_malformed_payload():
    resp = client.post("/analyze", json={"case_id": "X"})
    assert resp.status_code == 422

def test_analyze_returns_decision_response_on_valid_payload():
    fake = DecisionResponse(
        case_id="PUB-001", decision="ADMISSIBLE", confidence=0.9,
        key_findings=["ok"], applicable_limits=[], missing_evidence=[],
        citations=[], validation=ValidationResult(status="PASS", unsupported_claims=[]), trace=[],
    )
    valid_payload = {
        "case_id": "PUB-001", "policy_id": "P", "policy_start_date": "2025-01-01",
        "claim_date": "2026-01-01", "sum_insured_inr": 500000,
        "patient": {"age": 30}, "hospital": {"name": "H"},
        "treatment": {"type": "inpatient", "diagnosis": "flu"}, "task": "decide",
    }
    with patch("src.api.main.run_pipeline", return_value=fake):
        resp = client.post("/analyze", json=valid_payload)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "ADMISSIBLE"
```

- [ ] **Step 3: Run `pytest tests/test_api.py -v`** — expect PASS.

- [ ] **Step 4: Commit**

```bash
git add src/api/main.py tests/test_api.py
git commit -m "feat: FastAPI /analyze and /health endpoints with graceful error handling"
```

---

### Task 13: Streamlit frontend

**Files:**
- Create: `frontend/streamlit_app.py`

**Interfaces:**
- Consumes: `BACKEND_URL` env var, `POST /analyze` response shape from Task 12.
- Produces: a runnable `streamlit run frontend/streamlit_app.py` reviewer UI.

- [ ] **Step 1: Implement**

```python
import os
import json
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")
st.title("Policy-Aware Multi-Agent Claim Decision Engine")

STATUS_COLOR = {
    "ADMISSIBLE": "🟢", "ADMISSIBLE_WITH_LIMITS": "🔵",
    "PARTIALLY_ADMISSIBLE": "🟠", "NOT_ADMISSIBLE": "🔴", "NEEDS_REVIEW": "🟡",
}

@st.cache_data
def load_public_cases():
    path = os.environ.get("PUBLIC_CASES_PATH", "../Aptino_Candidate_Package_FINAL/candidate_data/public_test_cases.json")
    with open(path) as f:
        return json.load(f)

cases = load_public_cases()
case_ids = [c["case_id"] for c in cases]

with st.sidebar:
    st.header("Select a case")
    mode = st.radio("Input mode", ["Public test case", "Paste JSON"])
    if mode == "Public test case":
        selected_id = st.selectbox("Case", case_ids)
        case_payload = next(c for c in cases if c["case_id"] == selected_id)
        st.json(case_payload, expanded=False)
    else:
        raw = st.text_area("Paste claim case JSON", height=300)
        case_payload = json.loads(raw) if raw.strip() else None
    run = st.button("Run Analysis", type="primary", disabled=case_payload is None)

if run and case_payload is not None:
    with st.spinner("Running multi-agent analysis..."):
        resp = requests.post(f"{BACKEND_URL}/analyze", json=case_payload, timeout=120)
    if resp.status_code != 200:
        st.error(f"Request failed: {resp.status_code} — {resp.text}")
    else:
        result = resp.json()
        decision = result["decision"]
        st.subheader(f"{STATUS_COLOR.get(decision, '')} {decision}  (confidence: {result['confidence']:.2f})")

        if decision == "NEEDS_REVIEW":
            st.warning("⚠ The system abstained: evidence or required fields were insufficient for a safe decision.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Key findings**")
            for k in result["key_findings"]:
                st.write(f"- {k}")
        with col2:
            st.markdown("**Applicable limits**")
            for l in result["applicable_limits"]:
                st.write(f"- {l}")
        with col3:
            st.markdown("**Missing evidence**")
            for m in result["missing_evidence"]:
                st.write(f"- {m}")

        st.markdown("### Policy citations")
        if result["citations"]:
            st.table([
                {"Claim": c["claim"], "Page": c["page"], "Section": c["section"], "Chunk ID": c["chunk_id"]}
                for c in result["citations"]
            ])
        else:
            st.info("No citations returned for this decision.")

        st.markdown(f"### Validation: {result['validation']['status']}")
        for u in result["validation"]["unsupported_claims"]:
            st.write(f"- ⚠ {u}")

        with st.expander("Execution trace"):
            st.table([
                {"Agent": t["agent"], "Action": t["action"], "Elapsed (ms)": t["elapsed_ms"], "Detail": t["detail"]}
                for t in result["trace"]
            ])
```

- [ ] **Step 2: Manually verify** — run `python -m src.ingestion.build_index`, then `uvicorn src.api.main:app --reload` in one terminal and `streamlit run frontend/streamlit_app.py` in another; select a public case (e.g. `PUB-002`, the initial-waiting-period case) and confirm the decision, citations, and trace render correctly and a case designed to be ambiguous (insufficient evidence) shows the `NEEDS_REVIEW` banner.

- [ ] **Step 3: Commit**

```bash
git add frontend/streamlit_app.py
git commit -m "feat: Streamlit reviewer UI with decision, citations, and execution trace"
```

---

### Task 14: Custom test cases, evaluation harness, and failure analysis

**Files:**
- Create: `eval/custom_cases.json`, `eval/expected_outcomes.json`, `eval/gold_evidence.json`, `eval/run_eval.py`
- Create: `docs/architecture_note.md` (references failure analysis findings)

**Interfaces:**
- Consumes: `run_pipeline`, `public_test_cases.json` (read-only), `eval/custom_cases.json`.
- Produces: `python eval/run_eval.py` → `eval/results.json` + printed summary; must be runnable with zero manual steps beyond having the index built and an LLM key configured.

- [ ] **Step 1: Author `eval/custom_cases.json`** — at least 5 new cases, following `schema/claim_case_schema.md`, deliberately covering: (a) a clean day-care/<24h admission case, (b) a room-rent sub-limit breach, (c) a portability continuity case, (d) an irrelevant/noise attribute (e.g. an unrelated `patient.occupation` field) to test the "attribute not relevant to policy decision" reliability scenario, (e)+(f) two cases engineered to be genuinely ambiguous/under-evidenced (e.g. hospital bed-count not stated where the policy's hospital definition requires a minimum bed count, or a diagnosis with no matching policy clause) so they must resolve to `NEEDS_REVIEW`.

- [ ] **Step 2: Author `eval/expected_outcomes.json`** — for all 12 public + 5+ custom cases, hand-adjudicate against the actual policy text and record:

```json
{
  "PUB-001": {"expected_decision": "ADMISSIBLE_WITH_LIMITS", "rationale": "Appendectomy is standard inpatient treatment; room rent may exceed sub-limit per Section X.Y — verify against policy page/section once indexed.", "policy_reference": {"page": 0, "section": ""}},
  "PUB-002": {"expected_decision": "NOT_ADMISSIBLE", "rationale": "Claim falls within the 30-day initial waiting period (policy_start_date 2026-01-01, claim_date 2026-01-20)."}
}
```

(Every entry's `page`/`section` must be filled in against the real indexed chunks once Task 4 is built — this is the "how the expected outcome is established" documentation required by the assignment's evaluation section.)

- [ ] **Step 3: Author `eval/gold_evidence.json`** for a representative subset (at least the waiting-period and sub-limit cases): `{"PUB-002": {"question": "initial waiting period", "gold_chunk_ids": ["chunk-00xx"]}}`, filled in by inspecting `index_store/chunks.json` after Task 4.

- [ ] **Step 4: Write `eval/run_eval.py`**

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.case import ClaimCase
from src.orchestrator.pipeline import run_pipeline

PUBLIC_CASES_PATH = Path("../Aptino_Candidate_Package_FINAL/candidate_data/public_test_cases.json")
CUSTOM_CASES_PATH = Path(__file__).parent / "custom_cases.json"
EXPECTED_PATH = Path(__file__).parent / "expected_outcomes.json"
GOLD_EVIDENCE_PATH = Path(__file__).parent / "gold_evidence.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

def load_cases() -> list[dict]:
    public = json.loads(PUBLIC_CASES_PATH.read_text())
    custom = json.loads(CUSTOM_CASES_PATH.read_text())
    return public + custom

def citation_hit_rate(response) -> float:
    if not response.citations:
        return 0.0
    grounded = sum(1 for c in response.citations if c.page > 0 and c.chunk_id)
    return grounded / len(response.citations)

def recall_at_k(case_id: str, response, gold: dict) -> float | None:
    entry = gold.get(case_id)
    if not entry:
        return None
    retrieved_ids = {c.chunk_id for c in response.citations}
    gold_ids = set(entry["gold_chunk_ids"])
    if not gold_ids:
        return None
    return len(retrieved_ids & gold_ids) / len(gold_ids)

def main():
    cases = load_cases()
    expected = json.loads(EXPECTED_PATH.read_text())
    gold = json.loads(GOLD_EVIDENCE_PATH.read_text())

    results = []
    correct = 0
    needs_review_count = 0
    for raw in cases:
        case = ClaimCase.model_validate(raw)
        response = run_pipeline(case)
        exp = expected.get(case.case_id, {})
        is_correct = exp.get("expected_decision") == response.decision
        correct += int(is_correct)
        needs_review_count += int(response.decision == "NEEDS_REVIEW")
        results.append({
            "case_id": case.case_id,
            "decision": response.decision,
            "expected_decision": exp.get("expected_decision"),
            "correct": is_correct,
            "confidence": response.confidence,
            "citation_hit_rate": citation_hit_rate(response),
            "recall_at_k": recall_at_k(case.case_id, response, gold),
            "validation_status": response.validation.status,
        })

    summary = {
        "total_cases": len(cases),
        "accuracy": correct / len(cases),
        "needs_review_count": needs_review_count,
        "avg_citation_hit_rate": sum(r["citation_hit_rate"] for r in results) / len(results),
    }
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "cases": results}, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run `python eval/run_eval.py`**, confirm `eval/results.json` is produced and the printed summary shows `needs_review_count >= 2`.

- [ ] **Step 6: Write `docs/architecture_note.md`** (1-2 pages) covering: agent boundaries and why each is separate, state-flow diagram (text/ASCII is fine), retrieval design (hybrid + RRF + rerank rationale), abstention design, and at least 3 documented failure cases with root cause and the fix applied (e.g., "Initial chunker split a sub-limit clause from its heading, causing the Coverage Agent to miss the room-rent cap on PUB-00X — fixed by folding heading text into the first chunk of each section").

- [ ] **Step 7: Commit**

```bash
git add eval docs/architecture_note.md
git commit -m "feat: 5 custom test cases, hand-adjudicated expected outcomes, and reproducible eval harness"
```

---

### Task 15: Dockerfile, README, and deployment

**Files:**
- Create: `Dockerfile`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a container that builds the index at image-build time and serves the API; a README with local setup, API examples, and links to the deployed frontend/backend.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.ingestion.build_index
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Deploy backend to Render** (Docker web service, env vars from `.env.example` set in Render dashboard) and **frontend to Streamlit Community Cloud** pointing `BACKEND_URL` at the Render URL. Confirm `GET /health` returns `200` and a public-case run in the deployed Streamlit app returns a decision end-to-end.

- [ ] **Step 3: Write `README.md`** covering: architecture diagram (ASCII/embedded image), local setup (`pip install -r requirements.txt`, `python -m src.ingestion.build_index`, `uvicorn src.api.main:app`, `streamlit run frontend/streamlit_app.py`), `.env` variables, `POST /analyze` request/response example (use the `PUB-001` payload and a real response), `python eval/run_eval.py` instructions, known limitations (hand-curated section map doesn't generalize to arbitrary PDFs; keyword-overlap + single-model grounding check is a heuristic, not a full NLI system; retry loop capped at 1 for latency/cost), and live URLs.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile README.md
git commit -m "feat: Dockerfile and README with setup, API examples, and deployment links"
```
