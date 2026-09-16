# Architecture Overview

## What this system does

Given a structured health-insurance claim case (JSON) and the supplied policy
PDF (`USGIC-CSCIndividualHealthInsurance_2017-2018.pdf`), the system produces
a structured decision (`ADMISSIBLE` / `ADMISSIBLE_WITH_LIMITS` /
`PARTIALLY_ADMISSIBLE` / `NOT_ADMISSIBLE` / `NEEDS_REVIEW`) with citations
that trace back to a specific page/section/chunk of the policy, and an
execution trace that shows what each agent did without exposing any model's
internal chain-of-thought.

## Why five agents, and why they're separate processes rather than "one prompt asked five times"

Each agent has a **different input schema, a different output schema, and
(for two of them) a different tool it alone can call**. That's what makes
this "genuinely multi-agent" rather than the same context fanned out to
multiple LLM calls:

| Agent | Reads | Calls an LLM? | Produces | Can it see the raw policy text? |
|---|---|---|---|---|
| Case Analysis | raw `ClaimCase` | yes (dimension/question extraction only) | `CaseState` | No - it has never seen a policy chunk |
| Policy Evidence | `CaseState.checklist` | no | `EvidenceBundle` | Yes - it's the only agent that calls the retriever |
| Coverage & Exclusion | `CaseState` + `EvidenceBundle` | yes | `CoverageFindings` | Only the chunks Policy Evidence handed it |
| Decision | `CaseState` + `CoverageFindings` | yes (skipped entirely if abstaining) | `DraftDecision` | Only chunk_ids referenced in findings |
| Validation | `DraftDecision` + `EvidenceBundle` | yes (per-citation fact-check) | `ValidationResult` | Only the specific chunk each citation claims to cite |

Each arrow in the pipeline is a **typed Pydantic object**
(`src/schemas/agent_state.py`), never a raw string. A later agent literally
cannot receive a field an earlier agent didn't populate, which is what the
assignment means by "agents exchange structured state."

## Control flow

```
ClaimCase
   |
   v
[Case Analysis Agent] --------------------> CaseState
   (deterministic field checks in Python;      (missing_fields, decision_dimensions,
    LLM only for dimension/question extraction)   checklist of retrieval questions)
   |
   v
[Policy Evidence Agent] -------------------> EvidenceBundle
   (no LLM call - runs HybridRetriever            (ranked EvidenceChunks per question,
    once per checklist question)                   each with page/section/chunk_id)
   |
   v
[Coverage & Exclusion Agent] --------------> CoverageFindings
   (LLM judges each decision dimension            (status + confidence + evidence_chunk_ids
    using ONLY the retrieved evidence text)         per dimension; UNCLEAR if evidence is weak)
   |
   v
[Decision Agent] ---------------------------> DraftDecision
   (Python checks first: any missing_field           (decision, citations, key_findings,
    or UNCLEAR/low-confidence finding forces           applicable_limits, missing_evidence)
    NEEDS_REVIEW WITHOUT calling the LLM;
    LLM only combines findings when everything
    is resolved)
   |
   v
[Validation Agent] -------------------------> ValidationResult
   (checks every citation: does the chunk_id           (PASS/FAIL + unsupported_claims)
    exist? keyword overlap? LLM entailment check?)
   |
   +--(FAIL, retry_count < 1)--> back to [Decision Agent] with feedback
   |
   +--(FAIL again, or retry exhausted)--> force decision = NEEDS_REVIEW
   |
   v
DecisionResponse (the API contract)
```

This is implemented as a single readable function
(`src/orchestrator/pipeline.py::run_pipeline`), not a graph framework. With
exactly one conditional edge (retry-on-FAIL) in the whole flow, a hand-rolled
state machine is easier to read, test, and explain than a general-purpose
graph abstraction would be for five sequential steps - see
[ADR-1 in modules/orchestrator.md](modules/orchestrator.md) for the fuller
trade-off discussion.

## Why abstention is enforced in two independent places

1. **Before the LLM ever runs** (`decision_agent.py::_forced_needs_review`):
   if the Case Analysis Agent found a missing required field/document, or
   the Coverage Agent marked any dimension `UNCLEAR` or below a confidence
   floor (0.55), the Decision Agent returns `NEEDS_REVIEW` deterministically
   and never asks the LLM to decide. This removes an entire failure class -
   "the model confidently answers anyway" - structurally, not by prompting.
2. **After the LLM runs** (`validation_agent.py` + the retry loop in
   `pipeline.py`): even if the Decision Agent produces a confident-looking
   decision, every citation is checked against the actual retrieved chunk
   text. If a citation can't be grounded, the pipeline retries once with
   that feedback, and if it still can't be grounded, the decision is
   force-overwritten to `NEEDS_REVIEW` regardless of what the LLM said.

Both layers are described in detail in
[modules/agents.md](modules/agents.md).

## Retrieval design

Dense (Chroma + `bge-small-en-v1.5`) and sparse (BM25) retrieval run in
parallel per checklist question, are combined with Reciprocal Rank Fusion,
and only the fused top candidates are passed to a cross-encoder reranker
(`bge-reranker-base`) before being handed to the Coverage Agent. See
[modules/retrieval.md](modules/retrieval.md) for why each stage exists and
what it would miss without the others.

## Chunking design

The policy is chunked on numbered clause boundaries (not fixed character
windows), and each chunk is tagged with the nearest heading that precedes it
("Definitions", "Scope of Cover", "Exclusions", ...), detected directly in
the extracted text rather than guessed from page ranges - the real document
has headings landing mid-page, which a page-range map would get wrong. See
[modules/ingestion.md](modules/ingestion.md) for the actual heading text
this was built from and the specific bug this approach caught during
development.

## Known limitations (see docs/architecture_note.md for the full write-up)

- The heading-detection list in `policy_sections.py` is specific to this
  policy document's actual heading strings; a different policy PDF would
  need its own heading list (or a more general layout-detection approach).
- The Validation Agent's grounding check is a keyword-overlap heuristic plus
  a single LLM entailment call, not a dedicated NLI model - documented
  false-positive/negative risk in the failure analysis.
- The retry loop is capped at 1 to bound latency and LLM cost per request.
