from unittest.mock import patch

from src.agents.coverage_exclusion_agent import run_coverage_exclusion
from src.schemas.agent_state import CaseState, EvidenceBundle, InvestigationItem
from src.schemas.case import ClaimCase

CASE = ClaimCase.model_validate(
    {
        "case_id": "T-1",
        "policy_id": "P",
        "policy_start_date": "2025-01-01",
        "claim_date": "2026-01-01",
        "sum_insured_inr": 500000,
        "patient": {"age": 30},
        "hospital": {"name": "H"},
        "treatment": {"type": "inpatient", "diagnosis": "flu"},
        "task": "decide",
    }
)
CASE_STATE = CaseState(
    case=CASE,
    decision_dimensions=["waiting_period"],
    missing_fields=[],
    checklist=[InvestigationItem(dimension="waiting_period", question="q1")],
)
EVIDENCE = EvidenceBundle(per_question={"q1": []})


def test_coverage_exclusion_returns_structured_findings():
    trace = []
    with patch(
        "src.agents.coverage_exclusion_agent.chat_json",
        return_value={
            "findings": [
                {
                    "dimension": "waiting_period",
                    "status": "WITHIN_WAITING_PERIOD",
                    "explanation": "policy started 5 days ago",
                    "evidence_chunk_ids": ["c1"],
                    "confidence": 0.9,
                }
            ]
        },
    ):
        findings = run_coverage_exclusion(CASE_STATE, EVIDENCE, trace)
    assert findings.findings[0].status == "WITHIN_WAITING_PERIOD"
    assert trace[0].agent == "CoverageExclusionAgent"


def test_coverage_exclusion_falls_back_to_unclear_on_llm_error():
    from src.llm.client import LLMOutputError

    trace = []
    with patch("src.agents.coverage_exclusion_agent.chat_json", side_effect=LLMOutputError("x")):
        findings = run_coverage_exclusion(CASE_STATE, EVIDENCE, trace)
    assert findings.findings[0].status == "UNCLEAR"
    assert findings.findings[0].confidence == 0.0
