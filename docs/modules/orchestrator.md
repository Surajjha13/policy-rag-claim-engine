# Orchestrator module (`src/orchestrator/pipeline.py`)

## `run_pipeline(case: ClaimCase) -> DecisionResponse`

The only function this module exposes. Wires the five agents together in
the fixed order described in `docs/architecture.md`, manages the one
validation retry, and translates the final `PipelineState` into the API's
`DecisionResponse` contract.

## ADR-1: a plain Python function instead of LangGraph

The assignment's tech guidance lists LangGraph as a recommended option, and
it was considered. It was not used, for a specific reason: **this pipeline's
control flow is linear with exactly one conditional edge** (retry the
Decision Agent if Validation fails, at most once). A graph framework earns
its complexity when there are multiple conditional branches, cycles with
varying re-entry points, or a need to visualize/serialize the graph
structure for reasons beyond this one process. None of that is true here.

Concretely, the entire control flow fits in ~20 lines
(`run_pipeline`'s body) that read top-to-bottom like the diagram in
`docs/architecture.md` - anyone reviewing this code can verify the retry
logic is correct by reading one function, without first learning a graph
framework's node/edge/state-channel vocabulary. A hand-rolled state machine
was judged easier to test (see `tests/test_pipeline.py`, which mocks every
agent function directly by name) and easier to explain in a design review
than the equivalent LangGraph `StateGraph` would be for the same five nodes.

If a future requirement needed branching retrieval strategies per dimension,
parallel agent fan-out, or persisted/resumable runs across process
restarts, that would be the point to revisit this decision and adopt
LangGraph (or a similar framework) - the `PipelineState` Pydantic model is
already structured compatibly with that migration (it's essentially a
LangGraph-style shared state object already).

## `get_retriever()` - process-level memoization

`HybridRetriever()` loads two ML models (a `SentenceTransformer` embedder
and a `CrossEncoder` reranker) - not something to redo per request. It's
stored in a module-level `_retriever` variable and built lazily on first
use. `GET /health` calls this same function to verify the index/models are
actually loadable, which is also how it reports `"degraded"` instead of
crashing if the index hasn't been built yet.

## Why the retry happens here and not inside `validation_agent.py`

`run_validation` only checks and reports; it has no opinion on what to do
about a `FAIL`. Putting the retry-or-abstain decision in the orchestrator
keeps `validation_agent.py` a pure function of (decision, evidence) -> result,
which is what makes `tests/test_validation_agent.py` able to test it in
complete isolation from the retry policy.
