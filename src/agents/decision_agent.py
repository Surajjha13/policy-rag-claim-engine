"""Decision Agent.

Two-tier abstention design: before ever calling the LLM, deterministic
Python checks whether required fields are missing, or whether NOTHING the
Coverage Agent found is usable (every dimension came back UNCLEAR/low
confidence). Either case returns NEEDS_REVIEW immediately - the LLM never
even sees a case with zero usable evidence.

A case with a MIX of confident and unclear findings is deliberately NOT
force-abstained in Python: the Case Analysis Agent's checklist often
includes exploratory dimensions (documentation completeness, network
status, sub-limits, ...) alongside the one that actually settles the case
(e.g. "is this within the waiting period"). Hard-blocking on any single
unclear dimension - even an irrelevant one - was tried first and produced
a real failure: a claim that squarely falls within the 30-day waiting
period (a confident, decisive, citable finding) was forced to NEEDS_REVIEW
solely because an unrelated "is the hospital in-network" dimension came
back UNCLEAR. See docs/architecture_note.md's failure analysis for the
full writeup. Mixed cases are instead handed to the LLM, whose prompt
(DECISION_SYSTEM) explicitly instructs it to still abstain if a *required*
dimension - not just any dimension - is unresolved; the Validation Agent
remains the safety net if that judgment call cites something ungrounded.
"""

import json

from src.agents.base import timed
from src.llm.client import LLMOutputError, chat_json
from src.llm.prompts import DECISION_SYSTEM
from src.schemas.agent_state import CaseState, CoverageFindings, DraftDecision, EvidenceBundle
from src.schemas.decision import Citation, TraceEvent

CONFIDENCE_FLOOR = 0.55
UNRESOLVED_CHUNK_SECTION = "UNKNOWN (chunk_id not found in retrieved evidence)"


def _chunk_lookup(evidence: EvidenceBundle) -> dict[str, tuple[int, str]]:
    lookup = {}
    for chunks in evidence.per_question.values():
        for c in chunks:
            lookup[c.chunk_id] = (c.page, c.section)
    return lookup


def _resolve_citations(raw_citations: list[dict], evidence: EvidenceBundle) -> list[Citation]:
    """Fill page/section from the actual retrieved chunks rather than trusting
    the LLM to recall them - the LLM only needs to name which chunk_id
    supports which claim. A chunk_id that isn't in the evidence pool at all
    (invented, or copied from a stale run) resolves to an obviously-fake
    page/section so the Validation Agent's chunk-existence check still
    catches it downstream instead of silently accepting a wrong citation."""
    lookup = _chunk_lookup(evidence)
    resolved = []
    for raw in raw_citations:
        chunk_id = raw.get("chunk_id", "")
        page, section = lookup.get(chunk_id, (0, UNRESOLVED_CHUNK_SECTION))
        resolved.append(
            Citation(
                claim=raw.get("claim", ""),
                source="policy.pdf",
                page=page,
                section=section,
                chunk_id=chunk_id,
            )
        )
    return resolved


def _forced_needs_review(case_state: CaseState, coverage: CoverageFindings) -> DraftDecision | None:
    unclear = [f for f in coverage.findings if f.status == "UNCLEAR" or f.confidence < CONFIDENCE_FLOOR]
    nothing_usable = bool(coverage.findings) and len(unclear) == len(coverage.findings)
    if case_state.missing_fields or nothing_usable:
        reasons = list(case_state.missing_fields) + [f"{f.dimension}: {f.explanation}" for f in unclear]
        return DraftDecision(
            decision="NEEDS_REVIEW",
            confidence=0.4,
            key_findings=[f.explanation for f in coverage.findings if f.status != "UNCLEAR"],
            applicable_limits=[],
            missing_evidence=reasons,
            citations=[],
        )
    return None


def run_decision(
    case_state: CaseState,
    coverage: CoverageFindings,
    evidence: EvidenceBundle,
    trace: list[TraceEvent],
    feedback: list[str] | None = None,
) -> DraftDecision:
    forced = _forced_needs_review(case_state, coverage)
    if forced is not None:
        trace.append(
            TraceEvent(
                agent="DecisionAgent",
                action="abstain_missing_evidence",
                elapsed_ms=0.0,
                detail={"reasons": len(forced.missing_evidence)},
            )
        )
        return forced

    payload = {
        "case_facts": case_state.case.model_dump(mode="json"),
        "findings": [f.model_dump() for f in coverage.findings],
        "revision_feedback": feedback or [],
        "instructions": (
            'Return JSON: {"decision": str, "confidence": float, "key_findings": [str], '
            '"applicable_limits": [str], "missing_evidence": [str], '
            '"citations": [{"claim": str, "chunk_id": str}]}. '
            "Each citation's claim must be a full natural-language sentence stating the specific "
            "policy-supported conclusion (e.g. 'The claim falls within the 30-day initial waiting "
            "period.'), never a short label or dimension name. Only name a chunk_id that appears "
            "in evidence_chunk_ids of one of the findings above - do not include page or section, "
            "those are filled in automatically."
        ),
    }

    def _call():
        try:
            return chat_json(DECISION_SYSTEM, json.dumps(payload))
        except LLMOutputError:
            return {
                "decision": "NEEDS_REVIEW",
                "confidence": 0.3,
                "key_findings": [],
                "applicable_limits": [],
                "missing_evidence": ["Decision model output was unparseable."],
                "citations": [],
            }

    result = timed("DecisionAgent", "combine_findings", trace, _call)
    raw_citations = result.pop("citations", [])
    return DraftDecision(citations=_resolve_citations(raw_citations, evidence), **result)
