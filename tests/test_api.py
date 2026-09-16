from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.schemas.decision import DecisionResponse, ValidationResult

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_analyze_rejects_malformed_payload():
    resp = client.post("/analyze", json={"case_id": "X"})
    assert resp.status_code == 422


def test_analyze_returns_decision_response_on_valid_payload():
    fake = DecisionResponse(
        case_id="PUB-001",
        decision="ADMISSIBLE",
        confidence=0.9,
        key_findings=["ok"],
        applicable_limits=[],
        missing_evidence=[],
        citations=[],
        validation=ValidationResult(status="PASS", unsupported_claims=[]),
        trace=[],
    )
    valid_payload = {
        "case_id": "PUB-001",
        "policy_id": "P",
        "policy_start_date": "2025-01-01",
        "claim_date": "2026-01-01",
        "sum_insured_inr": 500000,
        "patient": {"age": 30},
        "hospital": {"name": "H"},
        "treatment": {"type": "inpatient", "diagnosis": "flu"},
        "task": "decide",
    }
    with patch("src.api.main.run_pipeline", return_value=fake):
        resp = client.post("/analyze", json=valid_payload)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "ADMISSIBLE"
