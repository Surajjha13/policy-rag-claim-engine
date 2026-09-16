"""Policy Evidence Agent.

No LLM call here at all - its whole job is to run the hybrid retriever once
per checklist question from the Case Analysis Agent and hand back ranked,
metadata-rich evidence. Keeping this agent LLM-free is a deliberate design
choice: retrieval quality should be measurable and reproducible
independent of any LLM's variance.
"""

from src.agents.base import timed
from src.retrieval.retriever import HybridRetriever
from src.schemas.agent_state import CaseState, EvidenceBundle
from src.schemas.decision import TraceEvent


def run_policy_evidence(
    case_state: CaseState, retriever: HybridRetriever, trace: list[TraceEvent]
) -> EvidenceBundle:
    per_question = {}
    for item in case_state.checklist:
        chunks, counts = timed(
            "PolicyEvidenceAgent",
            f"retrieve:{item.dimension}",
            trace,
            retriever.retrieve,
            item.question,
        )
        trace[-1].detail = counts
        per_question[item.question] = chunks
    return EvidenceBundle(per_question=per_question)
