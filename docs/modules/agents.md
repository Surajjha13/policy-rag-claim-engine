# Agents module (`src/agents/`)

Each file is one agent. Every agent takes typed Pydantic input, returns
typed Pydantic output (`src/schemas/agent_state.py`), and appends exactly
one (or a few) `TraceEvent`s via the shared `timed()` helper in `base.py`.

## `base.py` - `timed(agent_name, action, trace, fn, *args, detail=None)`

Every agent's LLM/retrieval call is wrapped in this instead of hand-rolling
`time.perf_counter()` in five different files. Guarantees the trace format
returned to the API/frontend is identical across all agents - if a sixth
agent were added later, it could not accidentally emit a differently-shaped
`TraceEvent`.

## `case_analysis_agent.py` - `run_case_analysis(case, trace) -> CaseState`

Deliberately split into a deterministic half and an LLM half:

- **`_detect_missing_fields`** (pure Python, no LLM): checks
  `treatment.admission_hours`, `treatment.pre_existing`, and whether any of
  `{claim_form, discharge_summary, itemized_bill}` are present. This never
  hallucinates and costs nothing - the input JSON is already structured, so
  there's no reason to ask an LLM whether a field is `None`.
- **`_llm_extract`** (one LLM call): the free-text parts of the case
  (`diagnosis`, `procedure`, `task`) genuinely need language understanding
  to turn into `decision_dimensions` (e.g. `waiting_period`,
  `sub_limits`, `hospital_definition`) and natural-language retrieval
  questions for the Policy Evidence Agent. If the LLM call fails to return
  parseable JSON (`LLMOutputError`), the agent falls back to a single
  generic `coverage_scope` question rather than crashing the request.

**Known simplification** (see the failure analysis in
`docs/architecture_note.md`): the current document check only verifies at
least one of the three required documents is present, not that the specific
document appropriate to the treatment type (`discharge_summary` for
inpatient vs. `procedure_record`/`doctor_certificate` for day-care and
domiciliary claims) is present. This was deliberately kept simple for the
first pass rather than hardcoding a document-type matrix; PUB-006 and
PUB-011's document gaps happen to still be caught because they're missing
`itemized_bill` and `discharge_summary` respectively, which the current
check does surface via the `evidence_context` unresolved-field check instead
(see the docstring in `_detect_missing_fields`).

## `policy_evidence_agent.py` - `run_policy_evidence(case_state, retriever, trace) -> EvidenceBundle`

No LLM call at all - it just calls `retriever.retrieve()` once per checklist
question and collects results into `per_question`. This agent is kept
LLM-free on purpose: retrieval quality should be measurable and reproducible
independent of any LLM's sampling variance, which is what lets
`eval/run_eval.py` compute a meaningful recall@k number.

## `coverage_exclusion_agent.py` - `run_coverage_exclusion(case_state, evidence, trace) -> CoverageFindings`

One LLM call, given the case facts, the missing-fields list, and every
retrieved chunk grouped by the question it answers. The system prompt
(`COVERAGE_EXCLUSION_SYSTEM` in `llm/prompts.py`) explicitly instructs the
model to judge each dimension **using only the provided evidence text** and
to mark a dimension `UNCLEAR` with confidence below 0.5 if the evidence
doesn't clearly settle it - this is the mechanism that produces the
`DimensionFinding.status == "UNCLEAR"` the Decision Agent later treats as a
hard abstention signal. On an unparseable LLM response, it falls back to one
`UNCLEAR` finding rather than guessing a status.

## `decision_agent.py` - `run_decision(case_state, coverage, evidence, trace, feedback=None) -> DraftDecision`

The most important file for the "appropriate abstention" rubric criterion.
Two-tier design:

1. **`_forced_needs_review`** runs *before* any LLM call. If
   `case_state.missing_fields` is non-empty, or *every* `CoverageFindings`
   finding is `UNCLEAR`/below `CONFIDENCE_FLOOR = 0.55` (nothing at all is
   usable), it returns a `NEEDS_REVIEW` `DraftDecision` immediately, with
   `missing_evidence` listing exactly which fields/dimensions triggered it.
   **The LLM never sees a case in this state.** A *mix* of confident and
   unclear findings deliberately does NOT force this path - an earlier
   version treated any single unclear dimension as disqualifying and it
   force-abstained a case that was actually decisively resolved by one
   confident dimension while an unrelated exploratory dimension was merely
   unclear (see `docs/architecture_note.md`, failure case 4).
2. Otherwise it calls the LLM (`DECISION_SYSTEM`) to combine findings into a
   final decision + citations, having been told to still abstain if a
   dimension *necessary to this case's conclusion* is unresolved - a
   judgment call the Python gate above can't make since it doesn't know
   which dimensions are load-bearing for a given case.
3. **`_resolve_citations`** - the LLM is only asked for `{claim, chunk_id}`
   per citation, never `page`/`section`; those are looked up from the real
   `EvidenceBundle` afterward. Asking the LLM to recall page numbers from
   memory was tried first and both crashed (a missing field is a required
   Pydantic field) and was a needless hallucination surface (see failure
   case 5). An unresolvable `chunk_id` degrades to a sentinel `page=0`
   citation instead of crashing, which the Validation Agent's existence
   check then flags.
4. The `feedback` parameter is only populated on the pipeline's one retry
   (see `orchestrator/pipeline.py`), letting a second attempt see exactly
   which claims the Validation Agent rejected the first time.

## `validation_agent.py` - `run_validation(decision, evidence, trace) -> ValidationResult`

Three-layer grounding check per citation, cheapest first, so an unnecessary
LLM call is never made for a citation that's already provably wrong:

1. **Chunk existence** - does `citation.chunk_id` even appear in the
   evidence pool the pipeline actually retrieved? Catches an invented
   chunk_id for free.
2. **Keyword overlap** (`_keyword_overlap_ok`) - do at least 15% of the
   claim's distinctive tokens (>3 characters, crudely plural-stemmed)
   appear in the cited chunk's text? Deliberately a low bar - its only job
   is rejecting an *obviously* unrelated chunk cheaply, not judging
   correctness (that's step 3's job). A stricter 30% threshold was tried
   first and rejected a well-grounded but naturally-paraphrased citation
   before the entailment check ever ran (failure case 6 in
   `docs/architecture_note.md`).
3. **LLM entailment** - only for claims that pass 1 and 2: a dedicated,
   conservative fact-check prompt (`VALIDATION_SYSTEM`) asks "does this
   exact text support this exact claim," instructed to answer `false` when
   unsure.

Any single failure adds to `unsupported_claims` and the whole result is
`FAIL`, which `orchestrator/pipeline.py` uses to retry the Decision Agent
once and then force `NEEDS_REVIEW` if it still fails.

**Why this three-layer design and not a single NLI model call:** the
cheap checks catch the failure modes that don't need semantic
understanding (invented IDs, wrong-topic citations) without spending
latency/cost on an LLM call for every citation. This is a heuristic, not a
dedicated entailment model - see the false-positive/negative risk noted in
`docs/architecture_note.md`.
