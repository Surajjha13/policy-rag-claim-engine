"""Structured state exchanged BETWEEN agents.

This module is the backbone of the "genuine multi-agent" requirement: every
agent consumes and produces one of these typed models instead of raw text,
so a later agent can never silently misread an earlier agent's output.
"""

from pydantic import BaseModel, Field

from src.schemas.case import ClaimCase
from src.schemas.decision import Citation, DecisionStatus, TraceEvent, ValidationResult


class InvestigationItem(BaseModel):
    dimension: str
    question: str
    required: bool = True


class CaseState(BaseModel):
    """Output of the Case Analysis Agent."""

    case: ClaimCase
    decision_dimensions: list[str]
    missing_fields: list[str]
    checklist: list[InvestigationItem]


class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    section: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


class EvidenceBundle(BaseModel):
    """Output of the Policy Evidence Agent: ranked chunks per checklist question."""

    per_question: dict[str, list[EvidenceChunk]]


class DimensionFinding(BaseModel):
    dimension: str
    status: str
    explanation: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float


class CoverageFindings(BaseModel):
    """Output of the Coverage & Exclusion Agent."""

    findings: list[DimensionFinding]


class DraftDecision(BaseModel):
    """Output of the Decision Agent, before/after validation."""

    decision: DecisionStatus
    confidence: float
    # A model omitting one of these list fields means "none", not a parse
    # failure - gpt-oss-20b was observed dropping applicable_limits and
    # missing_evidence entirely on a case with nothing to report for either,
    # which crashed pydantic construction with an uncaught ValidationError
    # (decision_agent.py's LLMOutputError handling doesn't catch that).
    key_findings: list[str] = Field(default_factory=list)
    applicable_limits: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    citations: list[Citation]


class PipelineState(BaseModel):
    """Full run state threaded through the orchestrator for one /analyze call."""

    case_state: CaseState | None = None
    evidence: EvidenceBundle | None = None
    coverage: CoverageFindings | None = None
    draft_decision: DraftDecision | None = None
    validation: ValidationResult | None = None
    trace: list[TraceEvent] = []
    retry_count: int = 0
