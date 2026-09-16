"""FastAPI app exposing the two required endpoints.

/analyze validates the payload against ClaimCase (422 on malformed input,
pydantic gives field-level errors for free) then runs the pipeline. Any
unexpected pipeline exception (LLM outage, index not built, etc.) is
caught at this boundary and converted into a NEEDS_REVIEW response instead
of a raw 500 - a claims reviewer should always get a safe, structured
answer, never a stack trace.
"""

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from src.orchestrator.pipeline import get_retriever, run_pipeline
from src.schemas.case import ClaimCase
from src.schemas.decision import DecisionResponse, ValidationResult

app = FastAPI(title="Aptino Policy-Aware Claim Decision Engine")


@app.get("/health")
def health():
    try:
        get_retriever()
        index_ready = True
    except Exception:
        index_ready = False
    return {"status": "ok" if index_ready else "degraded", "index_ready": index_ready}


@app.post("/analyze", response_model=DecisionResponse)
def analyze(payload: dict):
    try:
        case = ClaimCase.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    try:
        return run_pipeline(case)
    except Exception as e:
        return DecisionResponse(
            case_id=payload.get("case_id", "UNKNOWN"),
            decision="NEEDS_REVIEW",
            confidence=0.0,
            key_findings=[],
            applicable_limits=[],
            missing_evidence=[f"Internal pipeline error: {type(e).__name__}"],
            citations=[],
            validation=ValidationResult(status="FAIL", unsupported_claims=[]),
            trace=[],
        )
