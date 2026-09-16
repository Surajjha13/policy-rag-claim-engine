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
   deterministic Python gate: any missing field or `UNCLEAR`/low-confidence
   finding forces `NEEDS_REVIEW` *without calling the LLM at all*.
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

Per checklist question: dense (Chroma, `bge-small-en-v1.5`) and sparse
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
