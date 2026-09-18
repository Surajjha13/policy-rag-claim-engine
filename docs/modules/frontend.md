# Frontend module (`frontend/streamlit_app.py`)

A thin rendering layer over the pipeline's `DecisionResponse` - it has no
logic of its own beyond input selection and display. This is deliberate:
every fact a reviewer sees comes from that one contract, so the UI can
never show something the pipeline itself doesn't expose (in particular, it
cannot leak chain-of-thought, because the pipeline never produces any).

**In-process, not over HTTP.** Originally this called `POST /analyze` on
the separate FastAPI backend. It now calls `run_pipeline()` directly and
renders `.model_dump()` of the result - the contract boundary is identical
(same Pydantic model, same fields, same "can't show more than the
contract exposes" guarantee), only the transport changed. This was forced
by deployment: Streamlit Community Cloud runs exactly one
`streamlit run` process per app and cannot also host the FastAPI service
as its own reachable backend, so this frontend became the sole public
deployment (see the README's "Deployment" section for the resulting known
gap: no separate live backend URL). `src/api/main.py` is unchanged and
still the way to run this pipeline as an independently curl-able service
locally or via Docker - it just isn't what the deployed Streamlit app
talks to anymore. `ensure_index_built()` (`@st.cache_resource`) replaces
the Dockerfile's build-at-image-time step, since Streamlit Cloud has no
custom build-command hook to build the index ahead of the first request.

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
