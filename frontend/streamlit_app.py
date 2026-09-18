"""Reviewer-facing Streamlit UI.

Runs the pipeline in-process (Streamlit Community Cloud only runs a single
`streamlit run` process - it can't also host the separate FastAPI service
as its own reachable backend, so this is the sole public deployment; see
docs/modules/frontend.md). It still renders exactly the DecisionResponse
contract and nothing else, so it can never show a reviewer something the
pipeline itself doesn't expose (in particular: no hidden chain-of-thought,
only the trace's agent/action/elapsed_ms/detail fields) - the boundary
that used to be an HTTP call is now a single `.model_dump()` at the same
point. `src/api/main.py` (FastAPI) is unchanged and still the way to run
this pipeline as an independently curl-able service locally or in Docker.
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud's launcher only puts this file's own directory
# (frontend/) on sys.path, not the repo root - unlike `python -m streamlit
# run ...` invoked from the repo root, which happens to put the cwd there.
# The `src.*` imports below need the repo root explicitly, regardless of
# how the process was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.build_index import build_index
from src.orchestrator.pipeline import run_pipeline
from src.schemas.case import ClaimCase
from src.schemas.decision import DecisionResponse, ValidationResult

PUBLIC_CASES_PATH = os.environ.get(
    "PUBLIC_CASES_PATH",
    "./data/candidate_cases/public_test_cases.json",
)


@st.cache_resource
def ensure_index_built() -> None:
    if not (Path(settings.index_dir) / "faiss.index").exists():
        build_index()


def analyze(case_payload: dict) -> dict:
    try:
        case = ClaimCase.model_validate(case_payload)
    except Exception as e:
        return {"_error": f"Invalid case JSON: {e}"}

    try:
        response = run_pipeline(case)
    except Exception as e:
        response = DecisionResponse(
            case_id=case_payload.get("case_id", "UNKNOWN"),
            decision="NEEDS_REVIEW",
            confidence=0.0,
            key_findings=[],
            applicable_limits=[],
            missing_evidence=[f"Internal pipeline error: {type(e).__name__}"],
            citations=[],
            validation=ValidationResult(status="FAIL", unsupported_claims=[]),
            trace=[],
        )
    return response.model_dump()

st.set_page_config(page_title="Aptino Claim Decision Engine", layout="wide")
st.title("Policy-Aware Multi-Agent Claim Decision Engine")

STATUS_COLOR = {
    "ADMISSIBLE": "🟢",
    "ADMISSIBLE_WITH_LIMITS": "🔵",
    "PARTIALLY_ADMISSIBLE": "🟠",
    "NOT_ADMISSIBLE": "🔴",
    "NEEDS_REVIEW": "🟡",
}


@st.cache_data
def load_public_cases():
    with open(PUBLIC_CASES_PATH) as f:
        return json.load(f)


cases = load_public_cases()
case_ids = [c["case_id"] for c in cases]

with st.sidebar:
    st.header("Select a case")
    mode = st.radio("Input mode", ["Public test case", "Paste JSON"])
    if mode == "Public test case":
        selected_id = st.selectbox("Case", case_ids)
        case_payload = next(c for c in cases if c["case_id"] == selected_id)
        st.json(case_payload, expanded=False)
    else:
        raw = st.text_area("Paste claim case JSON", height=300)
        case_payload = json.loads(raw) if raw.strip() else None
    run = st.button("Run Analysis", type="primary", disabled=case_payload is None)

if run and case_payload is not None:
    with st.spinner("Running multi-agent analysis..."):
        ensure_index_built()
        result = analyze(case_payload)
    if "_error" in result:
        st.error(result["_error"])
    else:
        decision = result["decision"]
        st.subheader(f"{STATUS_COLOR.get(decision, '')} {decision}  (confidence: {result['confidence']:.2f})")

        if decision == "NEEDS_REVIEW":
            st.warning("⚠ The system abstained: evidence or required fields were insufficient for a safe decision.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Key findings**")
            for k in result["key_findings"]:
                st.write(f"- {k}")
        with col2:
            st.markdown("**Applicable limits**")
            for limit in result["applicable_limits"]:
                st.write(f"- {limit}")
        with col3:
            st.markdown("**Missing evidence**")
            for m in result["missing_evidence"]:
                st.write(f"- {m}")

        st.markdown("### Policy citations")
        if result["citations"]:
            st.table(
                [
                    {"Claim": c["claim"], "Page": c["page"], "Section": c["section"], "Chunk ID": c["chunk_id"]}
                    for c in result["citations"]
                ]
            )
        else:
            st.info("No citations returned for this decision.")

        st.markdown(f"### Validation: {result['validation']['status']}")
        for u in result["validation"]["unsupported_claims"]:
            st.write(f"- ⚠ {u}")

        with st.expander("Execution trace"):
            st.table(
                [
                    {"Agent": t["agent"], "Action": t["action"], "Elapsed (ms)": t["elapsed_ms"], "Detail": t["detail"]}
                    for t in result["trace"]
                ]
            )
