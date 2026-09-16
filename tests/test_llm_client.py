from unittest.mock import patch

from litellm.exceptions import RateLimitError

from src.llm.client import LLMOutputError, chat_json


def _fake_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_chat_json_parses_valid_json():
    with patch("src.llm.client.litellm.completion", return_value=_fake_response('{"a": 1}')):
        assert chat_json("sys", "user") == {"a": 1}


def test_chat_json_strips_markdown_fences():
    with patch(
        "src.llm.client.litellm.completion", return_value=_fake_response('```json\n{"a": 2}\n```')
    ):
        assert chat_json("sys", "user") == {"a": 2}


def test_chat_json_raises_on_garbage():
    with patch("src.llm.client.litellm.completion", return_value=_fake_response("not json at all")):
        try:
            chat_json("sys", "user")
            assert False, "expected LLMOutputError"
        except LLMOutputError:
            pass


def _rate_limit_error(message: str = "rate limited") -> RateLimitError:
    return RateLimitError(message=message, llm_provider="groq", model="test-model")


def test_chat_json_retries_on_rate_limit_then_succeeds():
    with (
        patch(
            "src.llm.client.litellm.completion",
            side_effect=[_rate_limit_error(), _fake_response('{"ok": true}')],
        ),
        patch("src.llm.client.time.sleep") as mock_sleep,
    ):
        result = chat_json("sys", "user")
    assert result == {"ok": True}
    assert mock_sleep.called


def test_chat_json_raises_llm_output_error_after_exhausting_retries():
    with (
        patch("src.llm.client.litellm.completion", side_effect=_rate_limit_error()) as mock_completion,
        patch("src.llm.client.time.sleep"),
    ):
        try:
            chat_json("sys", "user")
            assert False, "expected LLMOutputError"
        except LLMOutputError:
            pass
    from src.llm.client import MAX_RETRIES

    assert mock_completion.call_count == MAX_RETRIES


def test_wait_seconds_parses_provider_retry_hint_instead_of_guessing():
    from src.llm.client import _wait_seconds

    error = _rate_limit_error("Rate limit reached... Please try again in 5.07s.")
    assert _wait_seconds(error, attempt=0) == 5.57


def test_wait_seconds_falls_back_to_exponential_backoff_without_a_hint():
    from src.llm.client import _wait_seconds

    error = _rate_limit_error("connection reset, no retry hint here")
    assert _wait_seconds(error, attempt=2) == 2.0 * (2**2)
