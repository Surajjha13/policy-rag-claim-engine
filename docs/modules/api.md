# API module (`src/api/main.py`)

Two endpoints, per the assignment's minimum requirement. No auth, no
extra endpoints - the assignment explicitly rewards a smaller, well-tested
system over one with unrequested surface area.

## `GET /health`

Calls `get_retriever()` (the same memoized retriever the pipeline uses) and
reports `{"status": "ok", "index_ready": true}` if it loads without
exception, or `{"status": "degraded", "index_ready": false}` if it throws
(e.g. the index hasn't been built yet, or a model file failed to load).
This makes `/health` an actual readiness check, not just "the process is
alive" - a load balancer or deploy script can use it to know the service
can really serve `/analyze`, not just that uvicorn is up.

## `POST /analyze`

Two separate failure boundaries, deliberately different in kind:

1. **`ClaimCase.model_validate(payload)` raising `ValidationError`** -> HTTP
   `422` with Pydantic's field-level error detail. This is a client error
   (malformed/unsupported request shape) and should look like one.
2. **Any exception from `run_pipeline`** (LLM provider outage, index file
   missing, an unexpected bug) -> HTTP `200` with a `DecisionResponse` whose
   `decision` is `NEEDS_REVIEW` and `missing_evidence` names the exception
   type. This is deliberate, not an oversight: a claims reviewer using the
   Streamlit UI should always see a structured, actionable response, never
   a raw 500/stack trace, even when something upstream genuinely broke. The
   "graceful error handling" requirement is interpreted here as "never
   surface a crash to the reviewer," not "never let an error happen."

`payload: dict` (rather than `payload: ClaimCase` as the route's declared
type) is intentional: it lets the handler catch `ValidationError` itself and
return the 422 with Pydantic's detail, instead of letting FastAPI's default
validation error handler take over with a differently-shaped error body.

## Why no separate `/evaluate` endpoint

The assignment asks for "a script/command that runs the evaluation
end-to-end" (`eval/run_eval.py`), not an API surface for it. Evaluation
needs the hand-labeled `expected_outcomes.json`/`gold_evidence.json` files
that live in the repo, not in a request payload - a CLI script that reads
them directly is simpler than an endpoint that would need them uploaded or
baked into the container image either way.
