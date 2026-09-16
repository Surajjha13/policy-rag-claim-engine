from unittest.mock import patch

from src.agents.validation_agent import run_validation
from src.schemas.agent_state import DraftDecision, EvidenceBundle, EvidenceChunk
from src.schemas.decision import Citation

EVIDENCE = EvidenceBundle(
    per_question={
        "q1": [
            EvidenceChunk(
                chunk_id="c1",
                text="The waiting period for pre-existing diseases is 48 months.",
                page=10,
                section="Waiting Periods",
            )
        ]
    }
)


def test_validation_fails_on_unknown_chunk_id():
    decision = DraftDecision(
        decision="ADMISSIBLE",
        confidence=0.9,
        key_findings=[],
        applicable_limits=[],
        missing_evidence=[],
        citations=[Citation(claim="x", source="policy.pdf", page=1, section="s", chunk_id="does-not-exist")],
    )
    trace = []
    result = run_validation(decision, EVIDENCE, trace)
    assert result.status == "FAIL"


def test_validation_fails_on_low_keyword_overlap():
    decision = DraftDecision(
        decision="ADMISSIBLE",
        confidence=0.9,
        key_findings=[],
        applicable_limits=[],
        missing_evidence=[],
        citations=[
            Citation(
                claim="ambulance charges are covered up to five thousand rupees",
                source="policy.pdf",
                page=10,
                section="Waiting Periods",
                chunk_id="c1",
            )
        ],
    )
    trace = []
    result = run_validation(decision, EVIDENCE, trace)
    assert result.status == "FAIL"


def test_validation_passes_when_claim_matches_chunk_and_model_agrees():
    decision = DraftDecision(
        decision="ADMISSIBLE",
        confidence=0.9,
        key_findings=[],
        applicable_limits=[],
        missing_evidence=[],
        citations=[
            Citation(
                claim="waiting period for pre-existing diseases is 48 months",
                source="policy.pdf",
                page=10,
                section="Waiting Periods",
                chunk_id="c1",
            )
        ],
    )
    trace = []
    with patch("src.agents.validation_agent.chat_json", return_value={"supported": True}):
        result = run_validation(decision, EVIDENCE, trace)
    assert result.status == "PASS"
