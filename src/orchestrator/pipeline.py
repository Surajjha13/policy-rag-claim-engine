"""The orchestrator: a deterministic 5-step state machine, not a framework.

CaseAnalysis -> PolicyEvidence -> CoverageExclusion -> Decision -> Validation
                                                            ^            |
                                                            +--(FAIL)----+
                                                     (retried at most once)

A plain function (not LangGraph) was chosen because the control flow here
is simple, linear, and has exactly one conditional edge (retry-on-FAIL) -
a hand-rolled state machine is easier to read, test, and explain line by
line than a graph-framework abstraction would be for five sequential
nodes. `HybridRetriever` is expensive to construct (loads two ML models),
so it is built once and memoized in `get_retriever()`.
"""

from src.config import settings
from src.agents.case_analysis_agent import run_case_analysis
from src.agents.coverage_exclusion_agent import run_coverage_exclusion
from src.agents.decision_agent import run_decision
from src.agents.policy_evidence_agent import run_policy_evidence
from src.agents.validation_agent import run_validation
from src.retrieval.retriever import HybridRetriever
from src.schemas.agent_state import PipelineState
from src.schemas.case import ClaimCase
from src.schemas.decision import DecisionResponse

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def run_pipeline(case: ClaimCase) -> DecisionResponse:
    state = PipelineState()
    state.case_state = run_case_analysis(case, state.trace)
    state.evidence = run_policy_evidence(state.case_state, get_retriever(), state.trace)
    state.coverage = run_coverage_exclusion(state.case_state, state.evidence, state.trace)
    state.draft_decision = run_decision(state.case_state, state.coverage, state.evidence, state.trace)
    state.validation = run_validation(state.draft_decision, state.evidence, state.trace)

    while state.validation.status == "FAIL" and state.retry_count < settings.validation_fail_retry_limit:
        state.retry_count += 1
        state.draft_decision = run_decision(
            state.case_state,
            state.coverage,
            state.evidence,
            state.trace,
            feedback=state.validation.unsupported_claims,
        )
        state.validation = run_validation(state.draft_decision, state.evidence, state.trace)

    if state.validation.status == "FAIL":
        state.draft_decision.decision = "NEEDS_REVIEW"
        state.draft_decision.missing_evidence.append(
            "Validation could not confirm evidence support for the decision; escalated for manual review."
        )

    return DecisionResponse(
        case_id=case.case_id,
        decision=state.draft_decision.decision,
        confidence=state.draft_decision.confidence,
        key_findings=state.draft_decision.key_findings,
        applicable_limits=state.draft_decision.applicable_limits,
        missing_evidence=state.draft_decision.missing_evidence,
        citations=state.draft_decision.citations,
        validation=state.validation,
        trace=state.trace,
    )
