# Frontend module (`frontend/streamlit_app.py`)

A thin rendering layer over `POST /analyze` - it has no logic of its own
beyond input selection and display. This is deliberate: every fact a
reviewer sees comes from the `DecisionResponse` JSON contract, so the UI can
never show something the API itself doesn't expose (in particular, it
cannot leak chain-of-thought, because the API never sends any).

## Layout decisions

- **Sidebar: case selection.** Either pick one of the 12 supplied public
  cases (loaded from `PUBLIC_CASES_PATH`, read-only) or paste arbitrary
  case JSON. This matches the assignment's "select/upload or paste a claim
  case" requirement without needing a file-upload widget for the common
  case (reviewing the supplied set).
- **Decision header** shows the status with a color-coded emoji
  (`STATUS_COLOR`) and the numeric confidence together, so a reviewer
  doesn't have to cross-reference a legend to tell `NOT_ADMISSIBLE` from
  `NEEDS_REVIEW` at a glance.
- **Explicit abstention banner**: `if decision == "NEEDS_REVIEW": st.warning(...)`
  is a separate, unmissable UI element, not just the status badge - the
  assignment specifically asks the UI to "clearly show when the system
  abstains."
- **Three-column findings/limits/missing-evidence row** mirrors the API
  contract's three corresponding list fields directly - no reformatting or
  summarizing happens client-side.
- **Citations as a table** (claim / page / section / chunk ID) rather than
  inline text, so a reviewer can scan many citations at once and cross-check
  page numbers against the physical policy PDF quickly.
- **Execution trace inside a collapsed `st.expander`**: visible on demand
  (agent name, action, elapsed ms, detail dict) without cluttering the
  primary decision view for a reviewer who just wants the answer.

## Why Streamlit and not a custom React/JS frontend

The assignment explicitly recommends Streamlit and states UI polish matters
less than evidence grounding/retrieval/agent-boundary quality. A
data-table-and-form UI over a JSON API is exactly Streamlit's strength, and
building a custom SPA here would spend effort the rubric doesn't reward.
