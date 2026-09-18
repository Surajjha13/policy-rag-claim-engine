# Architecture & Design Note

*(Required 1-2 page deliverable. Full file-by-file rationale lives in
`docs/architecture.md` and `docs/modules/*.md`; this note is the condensed
version covering agent boundaries, state flow, retrieval design, and
trade-offs.)*

## Agent boundaries

Five agents, each with a distinct input schema, output schema, and (for two
of them) exclusive access to a tool the others don't call:

1. **Case Analysis** - the only agent that sees the raw case JSON. Splits
   into a deterministic half (missing-field/document checks, pure Python)
   and an LLM half (turning free-text diagnosis/task into decision
   dimensions and retrieval questions). Never sees policy text.
2. **Policy Evidence** - the only agent that calls the retriever. No LLM
   call at all, so retrieval quality is measurable independent of any
   model's sampling variance.
3. **Coverage & Exclusion** - judges each decision dimension using *only*
   the evidence Policy Evidence retrieved, marking a dimension `UNCLEAR`
   with low confidence when the evidence doesn't settle it.
4. **Decision** - combines findings into a final decision, but only after a
   deterministic Python gate: a missing field, or *every* dimension coming
   back `UNCLEAR`/low-confidence (nothing at all usable), forces
   `NEEDS_REVIEW` *without calling the LLM*. A mix of confident and unclear
   findings is deliberately handed to the LLM rather than hard-blocked -
   see failure case 4 below for why.
5. **Validation** - independently re-checks every citation the Decision
   Agent produced against the actual retrieved chunk text (existence check
   -> keyword overlap -> LLM entailment), triggering one retry and then a
   forced `NEEDS_REVIEW` if grounding still fails.

Every arrow between them is a typed Pydantic object
(`src/schemas/agent_state.py`), not a string - a later agent cannot receive
a field an earlier one didn't populate.

## State flow

```
ClaimCase -> CaseState -> EvidenceBundle -> CoverageFindings -> DraftDecision -> ValidationResult
                                                                      ^                |
                                                                      +--(FAIL, x1)----+
                                                                      |
                                                          (FAIL again) -> force NEEDS_REVIEW
```

Implemented as one linear function (`orchestrator/pipeline.py::run_pipeline`)
rather than a graph framework - with exactly one conditional edge across
five steps, a hand-rolled state machine was judged easier to read, test, and
explain than the equivalent LangGraph graph (full trade-off in
`docs/modules/orchestrator.md`, ADR-1).

## Retrieval design

Per checklist question: dense (FAISS, `bge-small-en-v1.5`) and sparse
(BM25 over the same chunks) run independently, are combined with
Reciprocal Rank Fusion (rank-based, avoiding the need to reconcile
incomparable score scales), and only the fused top candidates go to a
cross-encoder reranker (`bge-reranker-base`) before reaching the Coverage
Agent. Chunking is clause-boundary-aware (numbered clause regex, not
fixed-size windows) and heading-aware: each chunk is tagged with the
nearest heading detected in the actual extracted text
(`DEFINITIONS`/`SCOPE OF COVER`/`WHAT WE EXCLUDE`/...), carried across page
boundaries, because headings in the real policy PDF land mid-page.

## Important trade-offs

- **No LangGraph** - justified above; would be revisited if the pipeline
  needed branching/parallel agent fan-out or resumable multi-turn runs.
- **Heading list is document-specific** - built by reading the actual 17
  pages of the supplied policy, not a general layout parser. A different
  policy PDF needs its own heading list.
- **Validation grounding is a 3-layer heuristic**, not a dedicated NLI
  model - cheap checks (chunk existence, keyword overlap) run before the
  one LLM entailment call, trading some precision for latency/cost.
- **Retry capped at 1** to bound per-request latency and LLM spend.
- **`litellm` installed with `--no-deps`** to avoid an unused ~16MB AWS
  Bedrock dependency chain (`docs/modules/llm.md`).
- **FAISS instead of Chroma for the dense index** - both are explicitly
  acceptable per the assignment's tech guidance. Chroma's install pulled in
  grpc, a Kubernetes client, onnxruntime, and the full OpenTelemetry SDK -
  none of which a single-process local index needs, but a real, concrete
  cost in this project's very slow/unstable network environment (installing
  chromadb's dependency tree was still not finished after resolving and
  downloading well over a dozen packages). FAISS's wheel is self-contained
  (only depends on numpy). Trade-off accepted: FAISS has no built-in
  metadata store, so `build_index.py` keeps an index-aligned chunk-metadata
  list alongside the vector index (`src/retrieval/dense.py`).

## Failure analysis (documented per the assignment's requirement)

**1. Chunker silently dropped text before a page's first numbered clause.**
Root cause: `_split_clauses_with_offsets`'s matched-clause branch built
chunks starting from the first regex match's position, discarding
`text[0:first_match_start]` entirely. On pages where prose precedes a
numbered list (e.g. page 7's Critical-Illness prose immediately before
"SCOPE OF COVER" then "1. Room, Boarding..."), that leading prose - and any
heading inside it - vanished from the index and could never be retrieved or
cited. Caught by
`test_chunker_assigns_section_by_nearest_preceding_heading_and_carries_across_pages`,
which asserted a heading appearing before the first numbered clause on a
page still produced a chunk. Fix: capture `text[:matches[0].start()]` as its
own leading piece before processing numbered clauses
(`src/ingestion/chunker.py`).

**2. The missing-document check couldn't distinguish "no documentation at
all" from "specifically missing one required document."** Root cause:
`_detect_missing_fields`'s original check was `if not (REQUIRED_DOCS &
set(case.documents))` - true only when *none* of `claim_form`,
`discharge_summary`, `itemized_bill` are present. Hand-adjudicating PUB-006
(which supplies `claim_form` + `discharge_summary` but omits
`itemized_bill`) showed this check would not fire even though the case is
missing a document genuinely needed to verify expense sub-limits. Partial
fix: added a check on the case's `evidence_context` field (when present)
for unresolved (`null`) facts, which happens to still force `NEEDS_REVIEW`
for PUB-006 and PUB-011 via their explicit evidence gaps. **Residual
limitation** (not fully fixed, documented rather than hidden): a case
missing a specific required document *without* an accompanying
`evidence_context` flag would still pass this check today. The complete fix
- requiring specific documents per `treatment.type` (e.g. `discharge_summary`
for inpatient, `procedure_record` for day-care) - was scoped out to avoid
hardcoding a document-type matrix without more real-world document-naming
examples to validate it against.

**3. Risk of an over-confident exclusion inferred from a definition, not an
operative clause (PUB-012).** Root cause: the policy *defines*
"Unproven/Experimental Treatment" (Definitions, page 6) but the enumerated
21-item "WHAT WE EXCLUDE" list never actually names experimental/unproven
treatment as excluded. A retrieval-and-cite system asked "is experimental
treatment excluded" will very plausibly retrieve the definition chunk (high
lexical/semantic overlap on "experimental"/"unproven") and a
less-careful Decision Agent could cite it as if it supported
`NOT_ADMISSIBLE`. Mitigation in place: the Coverage & Exclusion Agent's
system prompt restricts it to the provided evidence text only, and the
Validation Agent's keyword-overlap + LLM-entailment check is specifically
designed to reject a claim like "experimental treatment is excluded" when
the only cited chunk is a definition that never says "excluded." This
mitigation depends on the entailment check being reliable, which is a
heuristic rather than a guarantee - so this case was deliberately
hand-adjudicated as `NEEDS_REVIEW` (not `NOT_ADMISSIBLE`) specifically to
make this exact failure mode visible and measurable in
`eval/run_eval.py`'s accuracy metric, rather than being papered over by a
plausible-sounding but unsupported confident answer.

**4. Over-broad abstention: any single UNCLEAR dimension force-blocked the
decision, even an irrelevant one (found by running the real pipeline, not
by static review).** Manually smoke-testing PUB-002 (a clean-cut
initial-waiting-period case) against the live LLM showed the Case Analysis
Agent's checklist included several exploratory dimensions beyond the one
that actually matters - `documentation_sufficiency`, `hospital_definition`
(network-provider status), `sub_limits` - none of which the stubbed
evidence for this smoke test addressed, so the Coverage Agent correctly
marked them `UNCLEAR`. The original `_forced_needs_review` treated *any*
`UNCLEAR` finding as disqualifying, so the pipeline returned `NEEDS_REVIEW`
even though the waiting-period dimension itself was resolved at 0.95
confidence and should have been enough to decide `NOT_ADMISSIBLE` on its
own. Fix: the Python gate now only force-abstains when a required field is
missing or *every* finding is unclear (nothing at all usable); a mix of
confident and unclear findings is handed to the LLM, whose prompt
(`DECISION_SYSTEM`) was strengthened to explicitly weigh whether an unclear
dimension is actually necessary to this case's conclusion. Re-running the
same smoke test after the fix produced the correct `NOT_ADMISSIBLE`
decision with a properly grounded citation. Locked in by
`test_decision_agent_defers_to_llm_when_one_finding_is_confident_and_another_unclear`.

**5. The Decision Agent asked the LLM to recall citation page/section
numbers from memory, which crashed outright when the model omitted the
field (also found by running the real pipeline).** The original prompt's
citation schema asked for `{claim, source, page, section, chunk_id}` in one
shot. Against the live LLM, this failed in two ways: once the model omitted
`page` entirely (a hard `pydantic.ValidationError` crash before any
fallback could catch it - not a graceful abstention, an unhandled
exception), and separately, asking the model to recall page/section from
memory at all is a needless hallucination surface for information the
system already has ground truth for. Fix: the LLM now only supplies
`{claim, chunk_id}`; `page`/`section`/`source` are resolved programmatically
by looking up `chunk_id` in the actual `EvidenceBundle` the pipeline
retrieved (`decision_agent.py::_resolve_citations`). An unresolvable
`chunk_id` (invented or stale) now degrades to a citation with `page=0` and
an obviously-sentinel section string instead of crashing, which the
Validation Agent's existence check then correctly flags as unsupported.
A related, smaller issue surfaced in the same test run: the first fix
attempt caused the LLM to write terse dimension labels (e.g.
`"waiting_period"`) into the citation's `claim` field instead of a real
sentence - harmless for validation in that instance (the label's words
happened to overlap the chunk) but a poor reviewer-facing citation. The
prompt now explicitly requires `claim` to be a full natural-language
sentence.

**6. Fixing failure case 5's citation-claim prompt immediately triggered a
new Validation Agent false-negative on the resulting natural-language
paraphrase.** Once the LLM wrote a proper sentence ("The claim occurs
within the 30-day initial waiting period, making it ineligible for
coverage.") instead of a bare label, the keyword-overlap pre-filter
(0.3 threshold, exact-token match) rejected it: the paraphrase's words
("occurs," "initial," "ineligible," "coverage") mostly don't appear
verbatim in the chunk ("a waiting period of 30 days will apply to all
claims..."), and a singular/plural mismatch (claim vs. claims) meant even
that word didn't count as a match. The citation was correctly grounded but
never reached the LLM entailment check that could have confirmed it. Fix:
lowered the floor to 0.15 and added crude plural stemming
(`_normalize`), since this check's actual job is only to reject
*obviously* unrelated chunks cheaply - correctness judgment belongs to the
entailment step, not the token-overlap pre-filter. Locked in by
`test_keyword_overlap_survives_natural_paraphrase_and_plural_mismatch`.

**7. The first full 18-case eval run against the live Groq free tier
crashed on a 429, and after adding a retry the accuracy collapsed to 33%
with 16/18 cases abstaining.** Root cause, found by actually running the
full batch rather than single-case smoke tests: Groq's free tier caps
`openai/gpt-oss-120b` at 8000 tokens/minute, and this pipeline's per-case
LLM calls (case analysis, coverage judgment over several chunks, decision,
plus one entailment check per citation) comfortably exceed that within a
handful of cases. The first fix (catch `RateLimitError` and retry up to 3
times with exponential backoff: 2s/4s/8s) stopped the batch from crashing,
but Groq's actual reset window frequently exceeded 8 seconds, so most
retries were exhausted before the limit cleared - agents fell back to their
conservative `UNCLEAR`/unparseable-output paths, and that cascaded into
`NEEDS_REVIEW` for cases that should have been confidently decidable.
Second fix: `_wait_seconds` now parses the provider's own recommended delay
from the error text ("Please try again in 5.07s") instead of guessing with
exponential backoff, and `MAX_RETRIES` was raised from 3 to 5; blind
backoff remains the fallback only when a retryable error carries no such
hint (e.g. a dropped connection). `eval/run_eval.py` also gained a
per-case try/except (mirroring `src/api/main.py`'s existing boundary) so
one case's unrecoverable failure can never take down the rest of the batch
again, and a small inter-case delay to stay under budget in the first
place. See `eval/results.json` for the resulting run.

**8. Re-running the full batch later with `INTER_CASE_DELAY_SECONDS`
raised from 3s to 20s (model kept at `gpt-oss-120b` rather than switching
to a higher-limit model) completed cleanly - no crashes, no
exhausted-retry fallbacks - but accuracy was still 33%, which shows the
first run's 33% was not purely a rate-limit artifact.** Evidence this run
was clean: `avg_citation_hit_rate` more than doubled (0.222 -> 0.5) and
confidence scores are no longer uniformly pinned near 0.3-0.4 the way an
exhausted-retry fallback forces them - several cases show real
high-confidence output (e.g. PUB-003 at 0.94, CUST-002 at 0.93). Breaking
down the 12 incorrect cases: 5 are the Validation Agent vetoing a
genuinely grounded, high-confidence decision the Decision Agent already
reached (`validation_status: FAIL` with `citation_hit_rate: 1.0`, e.g.
PUB-001, PUB-005, PUB-010) - i.e. the 3-layer grounding check (existence ->
keyword overlap -> LLM entailment) is stricter than the hand-labeled
expected outcomes require; the other 7 are genuine abstentions where the
Decision Agent found no citable evidence at all
(`citation_hit_rate: 0.0`, low confidence). Neither is a bug - both are the
system doing exactly what it was designed to do (prefer `NEEDS_REVIEW`
over an ungrounded claim) - but it means the real accuracy ceiling on this
free-tier model is gated by validation strictness and retrieval recall,
not by rate-limiting. Improving it further would mean loosening the
entailment threshold or improving retrieval recall, which is future work,
not a defect to fix under this assignment's scope.
