# Policy-Aware Multi-Agent RAG Claim Decision Engine

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-pytest-0a9edc)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A five-agent claims-decision system over a real health insurance policy
(Universal Sompo CSC Individual Health Insurance, `UNIHLIP18004V011718`),
built to *show its work*: every decision is backed by hybrid (dense + BM25)
retrieval with cross-encoder reranking, a structured inter-agent pipeline,
and a citation-validation gate that abstains (`NEEDS_REVIEW`) rather than
hallucinate a coverage answer the policy text doesn't actually support.

**Pipeline:** Case Analysis → Policy Evidence (retrieval) → Coverage &
Exclusion → Decision → Validation (with one bounded retry on failed
grounding checks).

See `docs/architecture.md` for the full design, `docs/modules/*.md` for a
file-by-file walkthrough of *why* each piece exists, and
`docs/architecture_note.md` for the required 1-2 page design note.

## Repository layout

```
src/
  schemas/       API contract + inter-agent state (Pydantic models)
  ingestion/     PDF parsing, heading-aware clause chunking, index build
  retrieval/     Dense (FAISS) + sparse (BM25) + RRF fusion + reranker
  llm/           Provider-agnostic LLM client + per-agent prompts
  agents/        Case Analysis, Policy Evidence, Coverage & Exclusion,
                 Decision, Validation
  orchestrator/  The 5-step pipeline + one bounded retry loop
  api/           FastAPI app (/analyze, /health)
frontend/        Streamlit reviewer UI
data/            Bundled verbatim copies of the supplied policy PDF and
                 public_test_cases.json, so the deployed image is
                 self-contained (originals remain in
                 ../Aptino_Candidate_Package_FINAL/, unmodified)
eval/            Custom test cases, hand-adjudicated expected outcomes,
                 gold retrieval labels, and the reproducible eval script
tests/           pytest suite (chunker, retrieval fusion, each agent with
                 a mocked LLM, the orchestrator retry loop, the API)
docs/            Architecture + per-module documentation
```

## Local setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
pip install --no-deps "litellm>=1.48,<2"   # see requirements.txt for why this is separate
cp .env.example .env            # then set LLM_API_KEY to a real key
python -m src.ingestion.build_index   # builds index_store/ (FAISS + BM25)
```

Run the backend:

```bash
uvicorn src.api.main:app --reload
```

Run the frontend (in a second terminal):

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

Run the tests:

```bash
pytest
```

Run the evaluation (requires the index to be built and an LLM key configured):

```bash
python eval/run_eval.py
```

This writes `eval/results.json` and prints a summary (accuracy, average
citation hit rate, recall@k where labeled, and the `NEEDS_REVIEW` count).

## Configuration (`.env`, see `.env.example`)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | Passed to `litellm.completion` - swap providers with zero code changes |
| `EMBEDDING_MODEL` | `sentence-transformers` model for dense retrieval (default: `BAAI/bge-small-en-v1.5`) |
| `RERANK_MODEL` | Cross-encoder reranker (default: `BAAI/bge-reranker-base`) |
| `INDEX_DIR` | Where the built FAISS index + BM25 pickle live |
| `POLICY_PDF_PATH` | Path to the policy PDF to ingest |
| `TOP_K_DENSE` / `TOP_K_SPARSE` / `TOP_K_RERANKED` | Retrieval fan-out at each stage |
| `VALIDATION_FAIL_RETRY_LIMIT` | How many times the Decision Agent retries after a failed validation (default 1) |

No secrets are committed; `.env` is gitignored.

## API

### `GET /health`

```json
{"status": "ok", "index_ready": true}
```

### `POST /analyze`

Request body: a claim case JSON matching `schema/claim_case_schema.md`
(unknown/extra fields are tolerated). Example using the supplied `PUB-002`
case (falls inside the 30-day initial waiting period):

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "case_id": "PUB-002",
  "policy_id": "USGIC-CSC-2017-2018",
  "policy_start_date": "2026-01-01",
  "claim_date": "2026-01-20",
  "sum_insured_inr": 500000,
  "continuous_coverage_months": 0,
  "prior_insurer_continuous_years": 0,
  "patient": {"age": 41},
  "hospital": {"name": "City Care Hospital", "network_provider": true},
  "treatment": {
    "type": "inpatient", "admission_hours": 96, "diagnosis": "Viral fever",
    "procedure": "Medical management", "pre_existing": false, "experimental": false
  },
  "expenses_inr": {"room": 20000, "doctor_fees": 20000, "medicines_diagnostics": 30000,
                    "pre_hospitalization": 0, "post_hospitalization": 0, "ambulance": 0},
  "documents": ["claim_form", "discharge_summary", "itemized_bill"],
  "task": "Determine whether the claim falls within the initial waiting period and what the appropriate decision should be."
}
EOF
```

Response shape (see `src/schemas/decision.py::DecisionResponse`):

```json
{
  "case_id": "PUB-002",
  "decision": "NOT_ADMISSIBLE",
  "confidence": 0.9,
  "key_findings": ["Claim falls within the 30-day initial waiting period; no waiver condition is met."],
  "applicable_limits": [],
  "missing_evidence": [],
  "citations": [
    {"claim": "A waiting period of 30 days will apply to all claims unless...",
     "source": "policy.pdf", "page": 9, "section": "Exclusions", "chunk_id": "chunk-0046"}
  ],
  "validation": {"status": "PASS", "unsupported_claims": []},
  "trace": [
    {"agent": "CaseAnalysisAgent", "action": "extract_dimensions_and_checklist", "elapsed_ms": 812.3, "detail": {}},
    {"agent": "PolicyEvidenceAgent", "action": "retrieve:waiting_period", "elapsed_ms": 145.1,
     "detail": {"dense_k": 10, "sparse_k": 10, "fused_k": 14, "reranked_k": 4}},
    {"agent": "CoverageExclusionAgent", "action": "assess_dimensions", "elapsed_ms": 950.7, "detail": {}},
    {"agent": "DecisionAgent", "action": "combine_findings", "elapsed_ms": 700.2, "detail": {}},
    {"agent": "ValidationAgent", "action": "check_citation_grounding", "elapsed_ms": 430.5, "detail": {"citations_checked": 1}}
  ]
}
```

Malformed input (e.g. missing required fields) returns HTTP `422` with
Pydantic's field-level error detail.

## Design decisions, trade-offs, and known limitations

See `docs/architecture.md` (system-level) and `docs/modules/*.md`
(file-by-file rationale). In short:

- **Hand-rolled orchestrator, not LangGraph** - the control flow is linear
  with exactly one conditional edge; a framework wasn't judged to earn its
  complexity here (`docs/modules/orchestrator.md`, ADR-1).
- **Heading-detection chunking is specific to this policy's actual heading
  text** (verified by reading all 17 pages), not a generic layout parser -
  a different policy PDF would need its own heading list
  (`docs/modules/ingestion.md`).
- **Validation Agent's grounding check is a 3-layer heuristic** (chunk
  existence -> keyword overlap -> LLM entailment), not a dedicated NLI
  model - documented false-positive/negative risk in
  `docs/architecture_note.md`.
- **Retry loop capped at 1** to bound latency/cost per request.
- **PUB-012 (experimental treatment) is labeled `NEEDS_REVIEW`, not
  `NOT_ADMISSIBLE`**, because the policy defines "Unproven/Experimental
  Treatment" but never actually excludes it in the enumerated exclusions
  list - see `docs/modules/evaluation.md` for the full reasoning.

## Deployment

- Backend: Docker image (see `Dockerfile`) deployable to Render/Fly/any
  container host that can run a FastAPI app; the index is built at image
  build time so cold start doesn't re-embed the policy.
- Frontend: Streamlit Community Cloud, pointed at the deployed backend via
  the `BACKEND_URL` environment variable.
- Live URLs: _to be filled in after deployment._
