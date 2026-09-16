"""Decision Agent.

Two-tier abstention design: before ever calling the LLM, deterministic
Python checks whether any decision-critical dimension is UNCLEAR/low
confidence, or a required field was flagged missing by the Case Analysis
Agent. If so it returns NEEDS_REVIEW immediately - the LLM never even sees
a case it structurally cannot safely decide, which removes an entire class
of "confident but wrong" failure. Only when every dimension is resolved
does the LLM combine findings into a final decision + citations.
"""

import json

from src.agents.base import timed
from src.llm.client import LLMOutputError, chat_json
from src.llm.prompts import DECISION_SYSTEM
from src.schemas.agent_state import CaseState, CoverageFindings, DraftDecision
from src.schemas.decision import TraceEvent

CONFIDENCE_FLOOR = 0.55


def _forced_needs_review(case_state: CaseState, coverage: CoverageFindings) -> DraftDecision | None:
    unclear = [f for f in coverage.findings if f.status == "UNCLEAR" or f.confidence < CONFIDENCE_FLOOR]
    if case_state.missing_fields or unclear:
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
            '"citations": [{"claim": str, "source": "policy.pdf", "page": int, '
            '"section": str, "chunk_id": str}]}'
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
    return DraftDecision(**result)
