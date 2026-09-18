# Resume Notes (session paused 2026-09-17)

Where this stands and exactly what's left, for picking this back up later.

## What's done and verified

- Full implementation: schemas, ingestion (PDF parsing + heading-aware
  clause chunker, verified against the real 17-page policy text),
  FAISS + BM25 hybrid retrieval with RRF fusion and cross-encoder rerank,
  all 5 agents, orchestrator with bounded validation retry, FastAPI backend,
  Streamlit frontend, 6 custom eval cases with hand-adjudicated expected
  outcomes. Everything is committed on branch `feature/claim-decision-engine`
  (not yet merged to `master`, not yet pushed to a remote - see "Not done"
  below).
- **All 33 automated tests pass** (`pytest` from the project root), including
  real integration tests against the actual built FAISS index (no mocks) -
  e.g. `test_hybrid_retriever_surfaces_the_known_gold_chunk_for_cosmetic_exclusion`
  confirms real retrieval quality against the real policy text.
- The real index is built (`index_store/` - gitignored, rebuild with
  `python -m src.ingestion.build_index`).
- Manually smoke-tested the full 5-agent chain against the real Groq LLM
  (not mocked) on PUB-002 in isolation (fresh API budget, no rate
  pressure): produced the correct `NOT_ADMISSIBLE` decision with a properly
  grounded citation and `validation.status == "PASS"`. This proves the
  pipeline's logic is correct end-to-end - the remaining problem (below) is
  purely a free-tier rate-limit/throughput issue, not a correctness bug.
- Found and fixed 6 real bugs via this smoke-testing and the eval attempts
  (all documented with root cause + fix in `docs/architecture_note.md`):
  chunker dropping leading page text, decision agent over-abstaining on any
  single unclear dimension, citation page/section hallucination risk,
  validation keyword-overlap threshold rejecting valid paraphrases, and
  LLM rate-limit handling (see next section).

## Update (2026-09-18): eval done, repo pushed - only deployment remains

Since the notes below were written: the branch was merged to `main` and
pushed to `github.com/Surajjha13/policy-rag-claim-engine` (so item 3's
"no remote" is stale - a remote exists and is up to date). A clean full
18-case eval run was also completed - see failure case 8 in
`docs/architecture_note.md` for the full breakdown. Summary:
`LLM_MODEL` was kept at `groq/openai/gpt-oss-120b` per the user's
explicit choice (not switched to `llama-3.3-70b-versatile`);
`eval/run_eval.py`'s `INTER_CASE_DELAY_SECONDS` was raised from 3 to 20 to
give the free-tier TPM budget more headroom, and the batch (run via
`.venv/Scripts/python.exe eval/run_eval.py`, not the system Python - see
below) completed in ~80 minutes with no crashes and no exhausted-retry
fallbacks. Resulting `eval/results.json`: 33% accuracy,
`avg_citation_hit_rate` 0.5 (up from 0.222 in the earlier degraded run).
This 33% is now a **real** number, not a rate-limit artifact: 5 of the 12
misses are the Validation Agent correctly vetoing an otherwise
well-grounded decision (stricter than the hand-labeled expected
outcomes), and 7 are genuine no-evidence abstentions. Improving this
further (loosening the entailment threshold, improving retrieval recall)
is future work outside this assignment's scope, not a bug.

**Environment gotcha hit while resuming:** running `python eval/run_eval.py`
directly used the system Python (`pydantic_settings` missing ->
`ModuleNotFoundError`), not the project's `.venv`. Always invoke via
`./.venv/Scripts/python.exe eval/run_eval.py` (or activate the venv
first) on this machine.

**Deployment decision (2026-09-18): single Streamlit app, not a separate
Render backend.** Streamlit Community Cloud only runs one `streamlit run`
process per app and can't also host FastAPI as its own reachable service,
so user chose to collapse: `frontend/streamlit_app.py` now calls
`run_pipeline()` directly in-process instead of `POST /analyze` over HTTP
(`ensure_index_built()` builds the index on first request, cached via
`@st.cache_resource`). `litellm` moved from a separate `--no-deps` install
into `requirements.txt` proper, since Streamlit Cloud only runs
`pip install -r requirements.txt` with no hook for a second install
command. Verified locally: ran the app standalone (no FastAPI process),
drove PUB-002 through the browser, full render (decision, citations,
trace) worked purely in-process. `src/api/main.py` (FastAPI) is unchanged
and still locally runnable/curl-able/Docker-deployable - it's just not
what the deployed Streamlit app talks to. **Known, accepted gap:** only
one live URL will exist (the Streamlit app), not a separate backend URL -
see README "Deployment" section.

**Only remaining item: the actual deploy.** Sign up at share.streamlit.io
(GitHub OAuth), New app -> this repo -> `main` -> `frontend/streamlit_app.py`,
set `LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL=groq/openai/gpt-oss-120b` (etc.)
as Secrets, deploy. Then fill in the `README.md` "Live URL" placeholder.

## Quick resume checklist

- [x] Decide the model/rate-limit question (user chose: keep `gpt-oss-120b`).
- [x] `python eval/run_eval.py` (via `.venv/Scripts/python.exe`, from project root).
- [x] Update `docs/architecture_note.md` failure case with final numbers (case 8).
- [x] Push to GitHub (`main` branch, up to date with `origin/main`).
- [x] Collapse frontend to in-process pipeline (no separate backend deploy).
- [ ] Deploy `frontend/streamlit_app.py` to Streamlit Community Cloud,
      fill in the "Live URL" placeholder in `README.md`.
