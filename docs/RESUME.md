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

## What's NOT done / blocked

**1. `eval/run_eval.py` has not produced a clean full 18-case
`eval/results.json` yet.** Root cause: `LLM_MODEL=groq/openai/gpt-oss-120b`
(set in the gitignored `.env`, matching the value the user supplied) has an
**8000 tokens/minute free-tier limit on Groq**, and this pipeline's
per-case LLM usage (case analysis + coverage judgment over multiple
evidence chunks + decision + one validation entailment call per citation)
routinely exceeds that within 1-2 cases. Retry/backoff
(`src/llm/client.py`) prevents crashes and parses Groq's actual
"try again in Xs" hint rather than guessing, but during this session the
account's rolling TPM budget was already heavily consumed by earlier
testing, so most retries were still exhausting before the window cleared -
producing artificial `NEEDS_REVIEW` fallbacks that don't reflect the
system's real decision quality (confirmed by the clean PUB-002 smoke test
above, run when the budget wasn't already exhausted).

Presented three options to the user; **awaiting their choice**:
- Switch `LLM_MODEL` to `groq/llama-3.3-70b-versatile` for the eval run
  (much higher free-tier TPM, likely to just work) - **recommended**.
- Keep `gpt-oss-120b` but pace every individual LLM call much further
  apart (not just between cases) - eval run would take 45-90+ minutes.
- User supplies a different/upgraded API key.

**Next step once decided:** update `.env`'s `LLM_MODEL` if switching, then
run `python eval/run_eval.py` from the project root (needs the index built
and enough free system memory - see note below). It writes
`eval/results.json` and prints a summary.

**2. This machine ran low on memory partway through today's session**
(unrelated to this project - other running applications), which killed
several background eval attempts mid-run via the harness's own protective
monitor (not a bug in this code - foreground execution of the same script,
tested directly, completed without issue). If resuming on a
memory-constrained machine again, run `eval/run_eval.py` in the foreground
rather than as a background task, or close other memory-heavy applications
first.

**3. Not deployed anywhere yet.** No GitHub remote is configured on this
repo (`git remote -v` is empty - it's local-only right now), and nothing
has been deployed to Render/Streamlit Cloud/etc. The assignment requires a
public GitHub repo, a live frontend URL, and a live backend URL. This needs
the user's own GitHub/hosting accounts - I can walk through the exact
`git push`/deploy steps once the user has (or wants me to help set up)
those accounts.

**4. `docs/architecture_note.md`'s failure-case write-up (case 7) should be
updated** once the eval actually completes cleanly, to reference the final
real `eval/results.json` numbers instead of the two provisional
single-case results gathered during debugging.

## Quick resume checklist

- [ ] Decide the model/rate-limit question above (or just try
      `llama-3.3-70b-versatile` - it's a safe default recommendation).
- [ ] `python eval/run_eval.py` (foreground, from project root).
- [ ] Update `docs/architecture_note.md` failure case 7 with final numbers.
- [ ] Create a GitHub repo, push this branch, open/merge a PR to `master`.
- [ ] Deploy backend (Render, Docker) and frontend (Streamlit Community
      Cloud), fill in the "Live URLs" placeholder in `README.md`.
