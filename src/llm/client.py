"""Provider-agnostic LLM JSON client.

Uses litellm so the model can be swapped (Groq, OpenAI, Anthropic, a local
Ollama model, ...) purely via environment variables (see .env.example) with
no code changes. Every agent call goes through chat_json() and gets back a
plain dict - never a hidden chain-of-thought string - because the system
prompt explicitly forbids prose and every agent prompt asks only for a
final structured answer.
"""

import json
import re
import time

import litellm
from litellm.exceptions import APIConnectionError, RateLimitError, ServiceUnavailableError, Timeout

from src.config import settings

RETRYABLE_ERRORS = (RateLimitError, ServiceUnavailableError, APIConnectionError, Timeout)
MAX_RETRIES = 5
BACKOFF_SECONDS = 2.0
RETRY_AFTER_PATTERN = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _wait_seconds(error: Exception, attempt: int) -> float:
    """Groq (and other OpenAI-compatible providers) put the exact
    recommended wait in the error body, e.g. "Please try again in 5.07s."
    Parsing that is far more efficient than blind exponential backoff: a
    full 18-case eval batch against Groq's free-tier tokens-per-minute
    limit was, in practice, retrying dozens of times per run, and guessing
    too short just re-hits the limit while guessing too long wastes the
    whole batch's runtime. Falls back to exponential backoff only when no
    hint is present (e.g. a connection drop, not a rate limit)."""
    match = RETRY_AFTER_PATTERN.search(str(error))
    if match:
        return float(match.group(1)) + 0.5
    return BACKOFF_SECONDS * (2**attempt)


class LLMOutputError(Exception):
    """Raised when the model did not return parseable JSON, or a transient
    provider error (rate limit, timeout, connection drop) persisted past
    every retry.

    Every agent that calls chat_json() catches this and falls back to a
    conservative, explicit NEEDS_REVIEW-leaning result rather than crashing
    the request or guessing. Running a free-tier provider (e.g. Groq) with a
    strict tokens-per-minute limit against a full 18-case eval batch is
    exactly the scenario this was built for - the first eval run against
    the live LLM crashed the whole batch on the first 429, which this
    retry/backoff (plus eval/run_eval.py's per-case try/except) fixes.
    """


def chat_json(system: str, user: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                messages=[
                    {
                        "role": "system",
                        "content": system
                        + "\nRespond with ONLY valid JSON. No prose, no markdown fences.",
                    },
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                timeout=30,
            )
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(_wait_seconds(e, attempt))
                continue
            raise LLMOutputError(
                f"LLM call failed after {MAX_RETRIES} attempts: {e}"
            ) from e

        content = response["choices"][0]["message"]["content"].strip()
        content = (
            content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMOutputError(
                f"Model did not return valid JSON: {content[:200]}"
            ) from e

    raise LLMOutputError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
