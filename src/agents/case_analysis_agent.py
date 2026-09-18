"""Case Analysis Agent.

Deliberately hybrid, not pure-LLM: the input JSON is already structured, so
checking for missing/insufficient fields (admission_hours, key documents,
pre_existing flag) is done in plain Python - deterministic, free, and never
hallucinated. The LLM is only asked to do the part that genuinely requires
language understanding: turning the free-text diagnosis/procedure/task
into a list of decision dimensions and natural-language retrieval
questions for the Policy Evidence Agent.
"""

import json

from src.agents.base import timed
from src.llm.client import LLMOutputError, chat_json
from src.llm.prompts import CASE_ANALYSIS_SYSTEM
from src.schemas.agent_state import CaseState, InvestigationItem
from src.schemas.case import ClaimCase
from src.schemas.decision import TraceEvent

REQUIRED_DOCS = {"claim_form", "discharge_summary", "itemized_bill"}


def _detect_missing_fields(case: ClaimCase) -> list[str]:
    missing = []
    if case.treatment.admission_hours is None:
        missing.append("treatment.admission_hours")
    if not (REQUIRED_DOCS & set(case.documents)):
        missing.append("documents (no claim_form/discharge_summary/itemized_bill present)")
    if case.treatment.pre_existing is None:
        missing.append("treatment.pre_existing")
    # `evidence_context` is an optional, un-declared field (see ClaimCase's
    # extra="allow") some cases use to flag facts the claimant/hospital
    # hasn't confirmed yet (e.g. {"hospital_registered": null}). Any key
    # left null there is evidence the case itself says is outstanding, so
    # it's treated the same as a missing structured field.
    evidence_context = getattr(case, "evidence_context", None)
    if isinstance(evidence_context, dict):
        unresolved = sorted(k for k, v in evidence_context.items() if v is None)
        if unresolved:
            missing.append(f"evidence_context unresolved: {', '.join(unresolved)}")
    return missing


def _llm_extract(case: ClaimCase) -> dict:
    user = json.dumps(
        {
            "case": case.model_dump(mode="json"),
            "instructions": (
                'Return JSON: {"decision_dimensions": [...], '
                '"checklist": [{"dimension": str, "question": str, "required": bool}]}'
            ),
        }
    )
    return chat_json(CASE_ANALYSIS_SYSTEM, user)


def run_case_analysis(case: ClaimCase, trace: list[TraceEvent]) -> CaseState:
    missing_fields = _detect_missing_fields(case)

    def _call():
        try:
            return _llm_extract(case)
        except LLMOutputError:
            return {
                "decision_dimensions": ["coverage_scope"],
                "checklist": [
                    {
                        "dimension": "coverage_scope",
                        "question": f"Is {case.treatment.diagnosis} covered under the policy?",
                        "required": True,
                    }
                ],
            }

    result = timed("CaseAnalysisAgent", "extract_dimensions_and_checklist", trace, _call)
    # .get(..., []) rather than result["..."]: a model dropping a whole key
    # (observed with gpt-oss-20b on other agents' outputs - see
    # DraftDecision/DimensionFinding) should fall through to downstream
    # agents finding nothing to work with and abstaining, not crash the
    # pipeline with an uncaught KeyError.
    checklist = [InvestigationItem(**item) for item in result.get("checklist", [])]
    return CaseState(
        case=case,
        decision_dimensions=result.get("decision_dimensions", []),
        missing_fields=missing_fields,
        checklist=checklist,
    )
