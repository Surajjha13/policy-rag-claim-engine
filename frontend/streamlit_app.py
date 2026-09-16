"""Reviewer-facing Streamlit UI.

Talks to the backend only over HTTP (BACKEND_URL) - it renders exactly the
DecisionResponse JSON contract and nothing else, so it can never show a
reviewer something the API itself doesn't expose (in particular: no hidden
chain-of-thought, only the trace's agent/action/elapsed_ms/detail fields).
"""

import json
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
PUBLIC_CASES_PATH = os.environ.get(
    "PUBLIC_CASES_PATH",
    "./data/candidate_cases/public_test_cases.json",
)

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
        resp = requests.post(f"{BACKEND_URL}/analyze", json=case_payload, timeout=120)
    if resp.status_code != 200:
        st.error(f"Request failed: {resp.status_code} — {resp.text}")
    else:
        result = resp.json()
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
