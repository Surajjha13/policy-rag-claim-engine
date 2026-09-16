from unittest.mock import patch

from src.orchestrator import pipeline
from src.schemas.agent_state import CaseState, CoverageFindings, DraftDecision, EvidenceBundle
from src.schemas.case import ClaimCase
from src.schemas.decision import ValidationResult

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


def test_pipeline_escalates_to_needs_review_after_exhausting_retry():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    draft = DraftDecision(
        decision="ADMISSIBLE", confidence=0.8, key_findings=["x"], applicable_limits=[], missing_evidence=[], citations=[]
    )
    fail = ValidationResult(status="FAIL", unsupported_claims=["x"])
    with (
        patch("src.orchestrator.pipeline.run_case_analysis", return_value=case_state),
        patch("src.orchestrator.pipeline.get_retriever", return_value=None),
        patch("src.orchestrator.pipeline.run_policy_evidence", return_value=EvidenceBundle(per_question={})),
        patch("src.orchestrator.pipeline.run_coverage_exclusion", return_value=CoverageFindings(findings=[])),
        patch("src.orchestrator.pipeline.run_decision", return_value=draft),
        patch("src.orchestrator.pipeline.run_validation", return_value=fail),
    ):
        response = pipeline.run_pipeline(CASE)
    assert response.decision == "NEEDS_REVIEW"
    assert response.validation.status == "FAIL"
    assert "escalated for manual review" in " ".join(response.missing_evidence)


def test_pipeline_returns_pass_result_without_retry_when_validation_passes():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    draft = DraftDecision(
        decision="ADMISSIBLE", confidence=0.9, key_findings=["x"], applicable_limits=[], missing_evidence=[], citations=[]
    )
    passed = ValidationResult(status="PASS", unsupported_claims=[])
    with (
        patch("src.orchestrator.pipeline.run_case_analysis", return_value=case_state),
        patch("src.orchestrator.pipeline.get_retriever", return_value=None),
        patch("src.orchestrator.pipeline.run_policy_evidence", return_value=EvidenceBundle(per_question={})),
        patch("src.orchestrator.pipeline.run_coverage_exclusion", return_value=CoverageFindings(findings=[])),
        patch("src.orchestrator.pipeline.run_decision", return_value=draft) as mock_decision,
        patch("src.orchestrator.pipeline.run_validation", return_value=passed),
    ):
        response = pipeline.run_pipeline(CASE)
    assert response.decision == "ADMISSIBLE"
    assert mock_decision.call_count == 1
