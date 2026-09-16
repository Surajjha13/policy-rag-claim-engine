from src.agents.policy_evidence_agent import run_policy_evidence
from src.schemas.agent_state import CaseState, EvidenceChunk, InvestigationItem
from src.schemas.case import ClaimCase


class FakeRetriever:
    def retrieve(self, query, top_k=None):
        chunk = EvidenceChunk(chunk_id="c1", text="waiting period is 30 days", page=10, section="Waiting Periods")
        return [chunk], {"dense_k": 5, "sparse_k": 5, "fused_k": 5, "reranked_k": 1}


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


def test_policy_evidence_agent_retrieves_per_checklist_item():
    case_state = CaseState(
        case=CASE,
        decision_dimensions=["waiting_period"],
        missing_fields=[],
        checklist=[InvestigationItem(dimension="waiting_period", question="what is the waiting period?")],
    )
    trace = []
    bundle = run_policy_evidence(case_state, FakeRetriever(), trace)
    assert "what is the waiting period?" in bundle.per_question
    assert bundle.per_question["what is the waiting period?"][0].chunk_id == "c1"
    assert trace[0].detail["reranked_k"] == 1
