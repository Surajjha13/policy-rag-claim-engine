"""Validation Agent.

Three-layer grounding check, cheapest-first:
1. Chunk-existence check - does the cited chunk_id even exist in the
   evidence pool the pipeline actually retrieved? Catches a model
   inventing a chunk_id.
2. Keyword-overlap check - do the claim's distinctive tokens (>3 chars,
   crudely stemmed) appear at all in the cited chunk text? This is
   deliberately a LOW bar (0.15), not a correctness check - its only job is
   to cheaply reject a citation pointing at an obviously unrelated chunk
   before spending an LLM call. Manual smoke-testing with the real LLM
   found that a stricter 0.3 threshold rejected a citation that was
   actually well-grounded but naturally paraphrased ("the claim occurs
   within the 30-day initial waiting period" vs. the chunk's "a waiting
   period of 30 days will apply to all claims") - it never even reached
   the entailment check in step 3. Precision belongs to step 3; this step
   only needs to catch the zero-overlap case.
3. LLM entailment check - only for claims that pass 1 and 2, ask a
   dedicated fact-check prompt "does this text really support this claim".
   Conservative by instruction: unsure -> not supported.

Any single failure marks that claim unsupported and the whole result FAILs,
which the orchestrator uses to retry the Decision Agent once (with the
failure list as feedback) and then force NEEDS_REVIEW if it still fails.
"""

import re

from src.agents.base import timed
from src.llm.client import LLMOutputError, chat_json
from src.llm.prompts import VALIDATION_SYSTEM
from src.schemas.agent_state import DraftDecision, EvidenceBundle
from src.schemas.decision import TraceEvent, ValidationResult

KEYWORD_OVERLAP_FLOOR = 0.15


def _all_chunks_by_id(evidence: EvidenceBundle) -> dict[str, str]:
    out = {}
    for chunks in evidence.per_question.values():
        for c in chunks:
            out[c.chunk_id] = c.text
    return out


def _normalize(token: str) -> str:
    # Crude plural stemming ("claims" -> "claim") - enough to stop an exact
    # singular/plural mismatch from masking real overlap, without pulling in
    # a full stemming library for what is only a cheap pre-filter.
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


def _keyword_overlap_ok(claim: str, chunk_text: str) -> bool:
    claim_tokens = {_normalize(t) for t in re.findall(r"[a-zA-Z0-9%]+", claim.lower()) if len(t) > 3}
    chunk_tokens = {_normalize(t) for t in re.findall(r"[a-zA-Z0-9%]+", chunk_text.lower())}
    if not claim_tokens:
        return True
    overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
    return overlap >= KEYWORD_OVERLAP_FLOOR


def run_validation(
    decision: DraftDecision, evidence: EvidenceBundle, trace: list[TraceEvent]
) -> ValidationResult:
    chunk_lookup = _all_chunks_by_id(evidence)
    unsupported: list[str] = []

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
                verdict = chat_json(
                    VALIDATION_SYSTEM, f"Claim: {citation.claim}\nPolicy text: {chunk_text}"
                )
                if not verdict.get("supported", False):
                    unsupported.append(f"model rejected support for claim: {citation.claim}")
            except LLMOutputError:
                unsupported.append(f"validation model failed to judge claim: {citation.claim}")
        return unsupported

    timed(
        "ValidationAgent",
        "check_citation_grounding",
        trace,
        _check_all,
        detail={"citations_checked": len(decision.citations)},
    )
    status = "FAIL" if unsupported else "PASS"
    return ValidationResult(status=status, unsupported_claims=unsupported)
