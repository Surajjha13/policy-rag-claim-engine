from unittest.mock import patch

from src.agents.decision_agent import run_decision
from src.schemas.agent_state import CaseState, CoverageFindings, DimensionFinding, EvidenceBundle, EvidenceChunk
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
EMPTY_EVIDENCE = EvidenceBundle(per_question={})


def test_decision_agent_forces_needs_review_when_missing_fields_present():
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=["documents missing"], checklist=[])
    coverage = CoverageFindings(findings=[])
    trace = []
    decision = run_decision(case_state, coverage, EMPTY_EVIDENCE, trace)
    assert decision.decision == "NEEDS_REVIEW"
    assert trace[0].action == "abstain_missing_evidence"


def test_decision_agent_forces_needs_review_when_every_finding_is_unclear():
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
    decision = run_decision(case_state, coverage, EMPTY_EVIDENCE, trace)
    assert decision.decision == "NEEDS_REVIEW"


def test_decision_agent_defers_to_llm_when_one_finding_is_confident_and_another_unclear():
    """Regression test: a claim squarely within the 30-day waiting period
    (confident, decisive) must not be force-abstained in Python just
    because an unrelated exploratory dimension (e.g. network-provider
    status) came back UNCLEAR - that was a real failure found via manual
    smoke-testing (see docs/architecture_note.md)."""
    case_state = CaseState(case=CASE, decision_dimensions=[], missing_fields=[], checklist=[])
    coverage = CoverageFindings(
        findings=[
            DimensionFinding(
                dimension="waiting_period",
                status="WITHIN_INITIAL_WAITING_PERIOD",
                explanation="claim made 19 days after policy inception, within the 30-day wait",
                evidence_chunk_ids=["chunk-0046"],
                confidence=0.95,
            ),
            DimensionFinding(
                dimension="hospital_definition",
                status="UNCLEAR",
                explanation="network-provider status not addressed by retrieved evidence",
                evidence_chunk_ids=[],
                confidence=0.2,
            ),
        ]
    )
    evidence = EvidenceBundle(
        per_question={
            "q1": [
                EvidenceChunk(chunk_id="chunk-0046", text="30 days waiting period...", page=9, section="Exclusions")
            ]
        }
    )
    trace = []
    with patch(
        "src.agents.decision_agent.chat_json",
        return_value={
            "decision": "NOT_ADMISSIBLE",
            "confidence": 0.9,
            "key_findings": ["Claim falls within the 30-day initial waiting period."],
            "applicable_limits": [],
            "missing_evidence": [],
            "citations": [
                {"claim": "Claim falls within the 30-day initial waiting period.", "chunk_id": "chunk-0046"}
            ],
        },
    ) as mock_chat:
        decision = run_decision(case_state, coverage, evidence, trace)
    assert mock_chat.called
    assert decision.decision == "NOT_ADMISSIBLE"
    assert decision.citations[0].page == 9
    assert decision.citations[0].section == "Exclusions"


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
    evidence = EvidenceBundle(
        per_question={"q1": [EvidenceChunk(chunk_id="c1", text="scope of cover text", page=7, section="Scope of Cover")]}
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
            "citations": [{"claim": "covered", "chunk_id": "c1"}],
        },
    ):
        decision = run_decision(case_state, coverage, evidence, trace)
    assert decision.decision == "ADMISSIBLE"
    assert decision.citations[0].chunk_id == "c1"
    assert decision.citations[0].page == 7
    assert decision.citations[0].section == "Scope of Cover"


def test_decision_agent_resolves_unknown_chunk_id_to_sentinel_instead_of_crashing():
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
            "citations": [{"claim": "covered", "chunk_id": "does-not-exist"}],
        },
    ):
        decision = run_decision(case_state, coverage, EMPTY_EVIDENCE, trace)
    assert decision.citations[0].page == 0
