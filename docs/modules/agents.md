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

## `decision_agent.py` - `run_decision(case_state, coverage, trace, feedback=None) -> DraftDecision`

The most important file for the "appropriate abstention" rubric criterion.
Two-tier design:

1. **`_forced_needs_review`** runs *before* any LLM call. If
   `case_state.missing_fields` is non-empty, or any `CoverageFindings`
   finding is `UNCLEAR` or below `CONFIDENCE_FLOOR = 0.55`, it returns a
   `NEEDS_REVIEW` `DraftDecision` immediately, with `missing_evidence`
   listing exactly which fields/dimensions triggered it. **The LLM never
   sees a case in this state** - abstention here is a structural guarantee,
   not a hope that the prompt worked.
2. Only when every dimension is resolved does it call the LLM
   (`DECISION_SYSTEM`) to combine findings into a final decision + citations.
   The prompt explicitly forbids inventing a `chunk_id`/page/number not
   present in the findings it was given - the Validation Agent is the
   actual enforcement of that, but stating it in the prompt reduces how
   often enforcement is needed.
3. The `feedback` parameter is only populated on the pipeline's one retry
   (see `orchestrator/pipeline.py`), letting a second attempt see exactly
   which claims the Validation Agent rejected the first time.

## `validation_agent.py` - `run_validation(decision, evidence, trace) -> ValidationResult`

Three-layer grounding check per citation, cheapest first, so an unnecessary
LLM call is never made for a citation that's already provably wrong:

1. **Chunk existence** - does `citation.chunk_id` even appear in the
   evidence pool the pipeline actually retrieved? Catches an invented
   chunk_id for free.
2. **Keyword overlap** (`_keyword_overlap_ok`) - do at least 30% of the
   claim's distinctive tokens (>3 characters) appear in the cited chunk's
   text? Catches a citation pointing at a real but irrelevant chunk without
   spending an LLM call.
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
