from unittest.mock import patch

from src.agents.case_analysis_agent import run_case_analysis
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
        "treatment": {"type": "inpatient", "diagnosis": "appendicitis", "admission_hours": 96},
        "documents": ["claim_form", "discharge_summary"],
        "task": "decide",
    }
)


def test_case_analysis_flags_missing_pre_existing_field():
    trace = []
    with patch(
        "src.agents.case_analysis_agent._llm_extract",
        return_value={
            "decision_dimensions": ["coverage_scope"],
            "checklist": [{"dimension": "coverage_scope", "question": "is it covered?", "required": True}],
        },
    ):
        state = run_case_analysis(CASE, trace)
    assert "treatment.pre_existing" in state.missing_fields
    assert len(trace) == 1
    assert trace[0].agent == "CaseAnalysisAgent"


def test_case_analysis_falls_back_on_llm_error():
    from src.llm.client import LLMOutputError

    trace = []
    with patch("src.agents.case_analysis_agent._llm_extract", side_effect=LLMOutputError("bad")):
        state = run_case_analysis(CASE, trace)
    assert state.decision_dimensions == ["coverage_scope"]


def test_case_analysis_flags_unresolved_evidence_context_fields():
    case_with_gap = ClaimCase.model_validate(
        {
            "case_id": "T-2",
            "policy_id": "P",
            "policy_start_date": "2025-01-01",
            "claim_date": "2026-01-01",
            "sum_insured_inr": 500000,
            "patient": {"age": 30},
            "hospital": {"name": "H"},
            "treatment": {
                "type": "inpatient",
                "diagnosis": "acute infection",
                "admission_hours": 96,
                "pre_existing": False,
            },
            "documents": ["claim_form", "discharge_summary"],
            "task": "decide",
            "evidence_context": {"hospital_registered": None, "medical_necessity_confirmed": None},
        }
    )
    trace = []
    with patch(
        "src.agents.case_analysis_agent._llm_extract",
        return_value={"decision_dimensions": ["coverage_scope"], "checklist": []},
    ):
        state = run_case_analysis(case_with_gap, trace)
    assert any("evidence_context unresolved" in m for m in state.missing_fields)
