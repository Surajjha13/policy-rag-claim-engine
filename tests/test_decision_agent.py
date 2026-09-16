from unittest.mock import patch

from src.agents.decision_agent import run_decision
from src.schemas.agent_state import CaseState, CoverageFindings, DimensionFinding
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


def test_decision_agent_forces_needs_review_when_missing_fields_present():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=["documents missing"], checklist=[])
    coverage = CoverageFindings(findings=[])
    trace = []
    decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "NEEDS_REVIEW"
    assert trace[0].action == "abstain_missing_evidence"


def test_decision_agent_forces_needs_review_when_finding_unclear():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    coverage = CoverageFindings(
        findings=[
            DimensionFinding(
                dimension="coverage_scope",
                status="UNCLEAR",
                explanation="no clause found",
                evidence_chunk_ids=[],
                confidence=0.2,
            )
        ]
    )
    trace = []
    decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "NEEDS_REVIEW"


def test_decision_agent_uses_llm_when_findings_are_confident():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    coverage = CoverageFindings(
        findings=[
            DimensionFinding(
                dimension="coverage_scope",
                status="COVERED",
                explanation="matches scope of cover",
                evidence_chunk_ids=["c1"],
                confidence=0.9,
            )
        ]
    )
    trace = []
    with patch(
        "src.agents.decision_agent.chat_json",
        return_value={
            "decision": "ADMISSIBLE",
            "confidence": 0.9,
            "key_findings": ["covered"],
            "applicable_limits": [],
            "missing_evidence": [],
            "citations": [
                {"claim": "covered", "source": "policy.pdf", "page": 7, "section": "Scope of Cover", "chunk_id": "c1"}
            ],
        },
    ):
        decision = run_decision(case_state, coverage, trace)
    assert decision.decision == "ADMISSIBLE"
    assert decision.citations[0].chunk_id == "c1"
