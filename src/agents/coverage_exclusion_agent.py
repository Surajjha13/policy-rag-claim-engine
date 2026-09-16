"""Coverage & Exclusion Agent.

Takes the case facts plus every retrieved chunk (grouped by the question it
answers) and asks the LLM to judge each decision dimension using ONLY that
evidence text - never general insurance knowledge (enforced by the system
prompt). Any dimension the model can't confidently settle from the
evidence is marked UNCLEAR with low confidence, which the Decision Agent
later treats as a hard signal to abstain.
"""

import json

from src.agents.base import timed
from src.llm.client import LLMOutputError, chat_json
from src.llm.prompts import COVERAGE_EXCLUSION_SYSTEM
from src.schemas.agent_state import CaseState, CoverageFindings, DimensionFinding, EvidenceBundle
from src.schemas.decision import TraceEvent


def _build_prompt_payload(case_state: CaseState, evidence: EvidenceBundle) -> dict:
    return {
        "case_facts": case_state.case.model_dump(mode="json"),
        "missing_fields": case_state.missing_fields,
        "evidence_by_question": {
            q: [{"chunk_id": c.chunk_id, "text": c.text, "page": c.page, "section": c.section} for c in chunks]
            for q, chunks in evidence.per_question.items()
        },
        "instructions": (
            'Return JSON: {"findings": [{"dimension": str, "status": str, '
            '"explanation": str, "evidence_chunk_ids": [str], "confidence": float}]}'
        ),
    }


def run_coverage_exclusion(
    case_state: CaseState, evidence: EvidenceBundle, trace: list[TraceEvent]
) -> CoverageFindings:
    def _call():
        try:
            payload = _build_prompt_payload(case_state, evidence)
            return chat_json(COVERAGE_EXCLUSION_SYSTEM, json.dumps(payload))
        except LLMOutputError:
            return {
                "findings": [
                    {
                        "dimension": "coverage_scope",
                        "status": "UNCLEAR",
                        "explanation": "Model output could not be parsed.",
                        "evidence_chunk_ids": [],
                        "confidence": 0.0,
                    }
                ]
            }

    result = timed("CoverageExclusionAgent", "assess_dimensions", trace, _call)
    return CoverageFindings(findings=[DimensionFinding(**f) for f in result["findings"]])
