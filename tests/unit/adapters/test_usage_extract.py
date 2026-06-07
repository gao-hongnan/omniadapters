"""Probe tests for the adapter usage-extraction constants.

They prove the provider/url/fallback/api_flavor constants each completion
adapter feeds to :meth:`pydantic_ai.usage.RequestUsage.extract` actually
populate token fields against the bundled genai-prices catalog.

These tests import ONLY ``pydantic_ai`` (and the provider SDK ``model_dump``
shapes the adapters produce). They deliberately do NOT import
``omniadapters.core.models`` so they can run while that module is being edited
in parallel.

Ground truth verified against the installed ``.venv`` sources:

- ``RequestUsage.extract`` never raises and never returns ``None``; on any
  failure it returns an all-zero ``RequestUsage`` (``has_values() is False``).
  So extraction correctness MUST be asserted via ``has_values()`` plus exact
  token fields, not via exceptions.
- The genai-prices ``openai`` provider has NO ``'default'`` api_flavor; its
  extractors are ``['chat', 'responses', 'embeddings']``. OpenAI/Azure must use
  ``api_flavor='chat'``. ``anthropic`` and ``google`` both expose ``'default'``.
- The genai-prices ``google`` extractor reads camelCase keys
  (``usageMetadata.promptTokenCount``, ``modelVersion``); the google-genai SDK
  ``model_dump`` is snake_case, so the gemini adapter dumps with
  ``by_alias=True``.
"""

from __future__ import annotations

from pydantic_ai.usage import RequestUsage

# --- OpenAI -----------------------------------------------------------------

# Shape of openai ChatCompletion.model_dump(mode="python") (relevant subset).
_OPENAI_DUMP = {
    "id": "chatcmpl-x",
    "model": "gpt-4o",
    "object": "chat.completion",
    "choices": [],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 10, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 8, "audio_tokens": 0},
    },
}


def test_openai_chat_flavor_extracts_tokens() -> None:
    usage = RequestUsage.extract(
        _OPENAI_DUMP,
        provider="openai",
        provider_url="https://api.openai.com",
        provider_fallback="openai",
        api_flavor="chat",
    )

    assert usage.has_values() is True
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cache_read_tokens == 10


def test_openai_default_flavor_yields_all_zero() -> None:
    """Guard that 'default' is rejected for openai and produces all-zero usage.

    'default' is NOT a valid openai api_flavor, so extraction silently produces
    an all-zero usage. This is why the adapter uses 'chat'.
    """
    usage = RequestUsage.extract(
        _OPENAI_DUMP,
        provider="openai",
        provider_url="https://api.openai.com",
        provider_fallback="openai",
        api_flavor="default",
    )

    assert usage.has_values() is False
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


# --- Anthropic --------------------------------------------------------------

# Shape of anthropic Message.model_dump(mode="python") (relevant subset).
_ANTHROPIC_DUMP = {
    "id": "msg_x",
    "model": "claude-3-5-sonnet-latest",
    "type": "message",
    "role": "assistant",
    "content": [],
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    },
}


def test_anthropic_default_flavor_extracts_tokens() -> None:
    usage = RequestUsage.extract(
        _ANTHROPIC_DUMP,
        provider="anthropic",
        provider_url="https://api.anthropic.com",
        provider_fallback="anthropic",
        api_flavor="default",
    )

    assert usage.has_values() is True
    # The anthropic catalog re-adds cached tokens onto input_tokens because the
    # SDK's input_tokens excludes cache read/creation: 100 + 10 + 5 == 115.
    assert usage.input_tokens == 115
    assert usage.output_tokens == 50
    assert usage.cache_read_tokens == 10
    assert usage.cache_write_tokens == 5


# --- Google / Gemini --------------------------------------------------------


def _gemini_dump_by_alias() -> dict[str, object]:
    """Build the camelCase dump the gemini adapter feeds to extract.

    Uses a real google-genai SDK response object dumped with ``by_alias=True``.
    """
    from google.genai.types import (
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )

    metadata = GenerateContentResponseUsageMetadata(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
        cached_content_token_count=10,
        thoughts_token_count=7,
    )
    response = GenerateContentResponse(model_version="gemini-2.5-flash", usage_metadata=metadata)
    return response.model_dump(mode="python", by_alias=True)


def test_google_default_flavor_extracts_tokens_from_by_alias_dump() -> None:
    usage = RequestUsage.extract(
        _gemini_dump_by_alias(),
        provider="google",
        provider_url="https://generativelanguage.googleapis.com",
        provider_fallback="google",
        api_flavor="default",
    )

    assert usage.has_values() is True
    assert usage.input_tokens == 100
    # output_tokens folds in thinking/thoughts tokens: 50 + 7 == 57.
    assert usage.output_tokens == 57
    assert usage.cache_read_tokens == 10


def test_google_snake_case_dump_yields_all_zero() -> None:
    """Guard that the snake_case dump produces all-zero usage for google.

    The snake_case model_dump (no by_alias) does NOT match the google catalog's
    camelCase key paths, so it produces an all-zero usage. This is why the
    gemini adapter must dump with by_alias=True.
    """
    snake_dump = {
        "model_version": "gemini-2.5-flash",
        "usage_metadata": {
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "total_token_count": 150,
        },
    }
    usage = RequestUsage.extract(
        snake_dump,
        provider="google",
        provider_url="https://generativelanguage.googleapis.com",
        provider_fallback="google",
        api_flavor="default",
    )

    assert usage.has_values() is False


# --- Azure (OpenAI-compatible) ----------------------------------------------


def test_azure_chat_flavor_extracts_tokens_with_endpoint() -> None:
    """Extract azure tokens using an azure endpoint URL.

    Azure carries the OpenAI ChatCompletion usage shape. With the azure provider
    id + an azure endpoint URL + 'openai' fallback + 'chat' flavor, extraction
    must populate tokens.
    """
    usage = RequestUsage.extract(
        _OPENAI_DUMP,
        provider="azure",
        provider_url="https://my-resource.openai.azure.com",
        provider_fallback="openai",
        api_flavor="chat",
    )

    assert usage.has_values() is True
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cache_read_tokens == 10


def test_azure_chat_flavor_extracts_tokens_with_fallback_url() -> None:
    """Extract azure tokens using the OpenAI fallback URL.

    When no azure endpoint is configured the adapter falls back to the OpenAI
    URL; extraction must still populate tokens.
    """
    usage = RequestUsage.extract(
        _OPENAI_DUMP,
        provider="azure",
        provider_url="https://api.openai.com",
        provider_fallback="openai",
        api_flavor="chat",
    )

    assert usage.has_values() is True
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
