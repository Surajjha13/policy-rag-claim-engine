from datetime import date

from pydantic import BaseModel, ConfigDict


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int


class Hospital(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    network_provider: bool | None = None


class Treatment(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    admission_hours: float | None = None
    diagnosis: str
    procedure: str | None = None
    pre_existing: bool | None = None
    experimental: bool | None = None


class ClaimCase(BaseModel):
    """Input contract for POST /analyze.

    ``extra="allow"`` (here and on the nested models) is deliberate: the
    assignment requires tolerating unknown/non-critical fields (e.g.
    ``prior_policy``, ``evidence_context``, ``expense_timing``, or an
    arbitrary attribute a reviewer adds) without rejecting the request.
    """

    model_config = ConfigDict(extra="allow")

    case_id: str
    policy_id: str
    policy_start_date: date
    claim_date: date
    sum_insured_inr: float
    continuous_coverage_months: int | None = None
    prior_insurer_continuous_years: int | None = None
    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: dict[str, float] = {}
    documents: list[str] = []
    task: str
