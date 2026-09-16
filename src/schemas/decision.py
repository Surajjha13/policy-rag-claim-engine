from typing import Literal

from pydantic import BaseModel

DecisionStatus = Literal[
    "ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS",
    "PARTIALLY_ADMISSIBLE",
    "NOT_ADMISSIBLE",
    "NEEDS_REVIEW",
]


class Citation(BaseModel):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str


class ValidationResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    unsupported_claims: list[str] = []


class TraceEvent(BaseModel):
    """One visible line of the execution trace.

    Deliberately excludes model reasoning/chain-of-thought: only agent name,
    the action taken, timing, and small structured counters (e.g. retrieval
    result counts) so the trace stays auditable without leaking hidden
    reasoning.
    """

    agent: str
    action: str
    elapsed_ms: float
    detail: dict = {}


class DecisionResponse(BaseModel):
    """The API's output contract - mirrors the assignment's example schema."""

    case_id: str
    decision: DecisionStatus
    confidence: float
    key_findings: list[str]
    applicable_limits: list[str]
    missing_evidence: list[str]
    citations: list[Citation]
    validation: ValidationResult
    trace: list[TraceEvent]
