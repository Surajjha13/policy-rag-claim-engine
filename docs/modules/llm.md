# LLM module (`src/llm/`)

**Install note:** `litellm` is installed with `--no-deps` (see
`requirements.txt`'s trailing comment). Its declared dependencies include
`boto3`/`botocore` for an AWS Bedrock integration this project never uses,
which added a slow ~16MB unnecessary download with no functional benefit for
our Groq/OpenAI-compatible usage. Every package `litellm` actually imports
at import time was found by iteratively running `import litellm` and
installing exactly what each successive `ModuleNotFoundError` named
(`openai`, `tiktoken`, `tokenizers`, `fastuuid`, `jinja2`, `jsonschema`,
`importlib-metadata`, `aiohttp`) - confirmed working by a clean
`import litellm` with none of the AWS-related packages present.

## `client.py` - `chat_json(system, user) -> dict`

The only function in the codebase that calls an LLM. Every agent goes
through this, never `litellm.completion` directly - which is what makes it
possible to swap providers/models via `.env` alone (`LLM_PROVIDER`,
`LLM_MODEL`, `LLM_API_KEY`) with zero code changes anywhere else.

Two things it enforces on every call:

1. **JSON-only, no chain-of-thought.** The system prompt passed in is always
   suffixed with `"Respond with ONLY valid JSON. No prose, no markdown
   fences."`, and the response is stripped of ` ```json ` fences before
   parsing. This is the mechanical enforcement behind "don't expose hidden
   chain-of-thought" - the model is never asked to narrate its reasoning in
   the first place, only to state a final structured answer.
2. **Fail loud, not silently wrong.** If the response still isn't valid JSON
   after fence-stripping, `chat_json` raises `LLMOutputError` rather than
   returning a partial/guessed dict. Every call site (`case_analysis_agent`,
   `coverage_exclusion_agent`, `decision_agent`, `validation_agent`) catches
   this specific exception and falls back to an explicit, conservative
   result (usually `UNCLEAR`/`NEEDS_REVIEW`-leaning), never to a silently
   empty or default-guessed structure.

`temperature=0.0` is hard-coded - these are structured extraction/judgment
calls, not creative generation, so determinism is preferred over sampling
diversity (and makes eval runs more reproducible run-to-run).

## `prompts.py` - one constant per agent

Kept as plain string constants rather than a templating engine because none
of them need runtime control flow (loops/conditionals) inside the prompt
text itself - the variable content (case facts, evidence, findings) is
always passed as the JSON `user` message, not interpolated into the system
prompt string. Each prompt ends with the same instruction pattern ("do not
explain your reasoning; output only the final JSON") for the reason above.

Two prompts are worth reading closely for the guarantees they encode:

- `DECISION_SYSTEM` explicitly says *"Never invent a chunk_id, page, or
  number not present in the findings"* - this is the first line of defense
  against fabricated citations, backed up by `validation_agent.py`'s
  chunk-existence check as the actual enforcement.
- `VALIDATION_SYSTEM` explicitly says *"Be conservative: if unsure, answer
  false"* - a fact-checker that defaults to rejecting is what makes the
  Validation Agent a meaningful gate rather than a rubber stamp.
