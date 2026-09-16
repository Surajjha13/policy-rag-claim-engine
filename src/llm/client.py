"""Provider-agnostic LLM JSON client.

Uses litellm so the model can be swapped (Groq, OpenAI, Anthropic, a local
Ollama model, ...) purely via environment variables (see .env.example) with
no code changes. Every agent call goes through chat_json() and gets back a
plain dict - never a hidden chain-of-thought string - because the system
prompt explicitly forbids prose and every agent prompt asks only for a
final structured answer.
"""

import json

import litellm

from src.config import settings


class LLMOutputError(Exception):
    """Raised when the model did not return parseable JSON.

    Every agent that calls chat_json() catches this and falls back to a
    conservative, explicit NEEDS_REVIEW-leaning result rather than crashing
    the request or guessing.
    """


def chat_json(system: str, user: str) -> dict:
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
