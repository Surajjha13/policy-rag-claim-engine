from src.schemas.case import ClaimCase


def test_claim_case_tolerates_unknown_fields():
    raw = {
        "case_id": "X-1",
        "policy_id": "P",
        "policy_start_date": "2025-01-01",
        "claim_date": "2026-01-01",
        "sum_insured_inr": 500000,
        "patient": {"age": 30, "gender": "F"},
        "hospital": {"name": "H"},
        "treatment": {"type": "inpatient", "diagnosis": "flu"},
        "task": "decide",
        "some_unmodeled_field": "ignore me",
    }
    case = ClaimCase.model_validate(raw)
    assert case.case_id == "X-1"
    assert case.patient.age == 30
