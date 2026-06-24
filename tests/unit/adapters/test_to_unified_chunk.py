"""Regression tests for ``_to_unified_chunk`` across every completion adapter.

These lock in the two correctness fixes that motivated the streaming revamp:

- **Anthropic** carries the real stop reason on the ``message_delta`` event
  (``delta.stop_reason``), *not* on ``message_stop`` (which the old code
  hard-coded to ``"stop"``). So ``tool_use`` must normalize to
  :attr:`FinishReason.TOOL_CALL`, and ``message_stop`` must yield no chunk.
- **Gemini**'s ``candidate.finish_reason`` is an enum; the old ``str(enum)``
  leaked the Python class name (``"FinishReason.STOP"``). It must normalize to
  :attr:`FinishReason.STOP`.

They also assert the normalized cross-provider shapes (``FinishReason`` enum +
``ToolCallDelta``) replacing the previous raw strings / ``dict[str, Any]``.

SDK chunk objects are built with ``model_validate`` so the tests exercise the
exact pydantic types the live APIs return.
"""

from __future__ import annotations

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
)
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletionChunk
from pydantic import SecretStr

from omniadapters.completion.adapters.anthropic import AnthropicAdapter
from omniadapters.completion.adapters.azure_openai import AzureOpenAIAdapter
from omniadapters.completion.adapters.gemini import GeminiAdapter
from omniadapters.completion.adapters.openai import OpenAIAdapter
from omniadapters.core.enums import FinishReason
from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    AzureOpenAICompletionClientParams,
    AzureOpenAIProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)


def _openai_adapter() -> OpenAIAdapter:
    return OpenAIAdapter(
        provider_config=OpenAIProviderConfig(api_key=SecretStr("k")),
        completion_params=OpenAICompletionClientParams(model="gpt-4o"),
    )


def _azure_adapter() -> AzureOpenAIAdapter:
    return AzureOpenAIAdapter(
        provider_config=AzureOpenAIProviderConfig(api_key=SecretStr("k")),
        completion_params=AzureOpenAICompletionClientParams(model="gpt-4o"),
    )


def _anthropic_adapter() -> AnthropicAdapter:
    return AnthropicAdapter(
        provider_config=AnthropicProviderConfig(api_key=SecretStr("k")),
        completion_params=AnthropicCompletionClientParams(model="claude-sonnet-4-6"),
    )


def _gemini_adapter() -> GeminiAdapter:
    return GeminiAdapter(
        provider_config=GeminiProviderConfig(api_key=SecretStr("k")),
        completion_params=GeminiCompletionClientParams(model="gemini-3-pro"),
    )


def _openai_chunk(choice: dict[str, object]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o", "choices": [choice]}
    )


# --- OpenAI / Azure (shared OpenAICompatibleAdapter) ------------------------


def test_openai_text_delta_yields_content() -> None:
    chunk = _openai_chunk({"index": 0, "delta": {"content": "hi"}, "finish_reason": None})
    result = _openai_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.content == "hi"
    assert result.finish_reason is None
    assert result.tool_calls is None
    assert result.model == "gpt-4o"


def test_openai_tool_call_delta_becomes_typed_tool_call() -> None:
    chunk = _openai_chunk(
        {
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": '{"a":'}}]},
            "finish_reason": None,
        }
    )
    result = _openai_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.tool_calls is not None
    (call,) = result.tool_calls
    assert call.index == 0
    assert call.id == "call_1"
    assert call.name == "f"
    assert call.args_json == '{"a":'


def test_openai_finish_reason_tool_calls_normalizes_to_tool_call() -> None:
    chunk = _openai_chunk({"index": 0, "delta": {}, "finish_reason": "tool_calls"})
    result = _openai_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.TOOL_CALL


def test_openai_empty_delta_yields_none() -> None:
    chunk = _openai_chunk({"index": 0, "delta": {}, "finish_reason": None})
    assert _openai_adapter()._to_unified_chunk(chunk) is None


def test_openai_no_choices_yields_none() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o", "choices": []}
    )
    assert _openai_adapter()._to_unified_chunk(chunk) is None


def test_azure_inherits_openai_finish_reason_normalization() -> None:
    # Proves AzureOpenAIAdapter reuses the shared OpenAICompatibleAdapter logic.
    chunk = _openai_chunk({"index": 0, "delta": {}, "finish_reason": "length"})
    result = _azure_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.LENGTH


def test_openai_function_call_finish_reason_aliases_to_tool_call() -> None:
    # The legacy `function_call` stop reason collapses onto TOOL_CALL like `tool_calls`.
    chunk = _openai_chunk({"index": 0, "delta": {}, "finish_reason": "function_call"})
    result = _openai_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.TOOL_CALL


def test_openai_content_filter_finish_reason_normalizes() -> None:
    chunk = _openai_chunk({"index": 0, "delta": {}, "finish_reason": "content_filter"})
    result = _openai_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.CONTENT_FILTER


# --- Anthropic --------------------------------------------------------------


def test_anthropic_message_delta_tool_use_maps_to_tool_call() -> None:
    # The real stop reason lives on message_delta, not message_stop.
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.finish_reason is FinishReason.TOOL_CALL


def test_anthropic_message_delta_end_turn_maps_to_stop() -> None:
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.finish_reason is FinishReason.STOP


def test_anthropic_message_stop_yields_no_chunk() -> None:
    # Old code hard-coded finish_reason="stop" here; it must now be inert.
    event = RawMessageStopEvent.model_validate({"type": "message_stop"})
    assert _anthropic_adapter()._to_unified_chunk(event) is None


def test_anthropic_text_delta_yields_content() -> None:
    event = RawContentBlockDeltaEvent.model_validate(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.content == "hi"


def test_anthropic_input_json_delta_becomes_tool_call() -> None:
    event = RawContentBlockDeltaEvent.model_validate(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.tool_calls is not None
    (call,) = result.tool_calls
    assert call.args_json == '{"a":'
    assert call.id is None
    assert call.name is None


def test_anthropic_tool_use_block_start_carries_id_and_name() -> None:
    event = RawContentBlockStartEvent.model_validate(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
        }
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.tool_calls is not None
    (call,) = result.tool_calls
    assert call.id == "toolu_1"
    assert call.name == "f"


def test_anthropic_message_delta_max_tokens_maps_to_length() -> None:
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.finish_reason is FinishReason.LENGTH


def test_anthropic_message_delta_refusal_maps_to_content_filter() -> None:
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "refusal", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )
    result = _anthropic_adapter()._to_unified_chunk(event)
    assert result is not None
    assert result.finish_reason is FinishReason.CONTENT_FILTER


def test_anthropic_message_delta_none_stop_reason_yields_no_chunk() -> None:
    # In-flight message_delta events carry no stop reason yet; they must be inert.
    event = RawMessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": None, "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }
    )
    assert _anthropic_adapter()._to_unified_chunk(event) is None


# --- Gemini -----------------------------------------------------------------


def test_gemini_finish_reason_enum_normalizes_not_stringified() -> None:
    # Old code did str(enum) -> "FinishReason.STOP"; must normalize to the enum.
    chunk = GenerateContentResponse.model_validate(
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finish_reason": "STOP"}]}
    )
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.content == "hi"
    assert result.finish_reason is FinishReason.STOP


def test_gemini_function_call_becomes_typed_tool_call() -> None:
    chunk = GenerateContentResponse.model_validate(
        {"candidates": [{"content": {"parts": [{"function_call": {"name": "f", "args": {"x": 1}}}]}}]}
    )
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.tool_calls is not None
    (call,) = result.tool_calls
    assert call.name == "f"
    # Gemini hands back already-parsed dict args (``function_call.args``); they
    # land on ``ToolCallDelta.args``, not ``args_json`` (the OpenAI/Anthropic
    # string-fragment field). Mirrors pydantic_ai, which passes the dict through.
    assert call.args == {"x": 1}


def test_gemini_no_candidates_yields_none() -> None:
    chunk = GenerateContentResponse.model_validate({"candidates": []})
    assert _gemini_adapter()._to_unified_chunk(chunk) is None


def test_gemini_safety_finish_reason_maps_to_content_filter() -> None:
    chunk = GenerateContentResponse.model_validate({"candidates": [{"finish_reason": "SAFETY"}]})
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.CONTENT_FILTER


def test_gemini_malformed_function_call_maps_to_error() -> None:
    # A finish-reason-only chunk (no content/tool calls) with a mapped reason
    # must still be emitted, carrying the normalized terminal signal.
    chunk = GenerateContentResponse.model_validate({"candidates": [{"finish_reason": "MALFORMED_FUNCTION_CALL"}]})
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.content == ""
    assert result.finish_reason is FinishReason.ERROR


def test_gemini_image_prohibited_content_maps_to_content_filter() -> None:
    # google-genai 2.3.0 added IMAGE_PROHIBITED_CONTENT; mirror pydantic_ai's map.
    chunk = GenerateContentResponse.model_validate({"candidates": [{"finish_reason": "IMAGE_PROHIBITED_CONTENT"}]})
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.CONTENT_FILTER


def test_gemini_no_image_finish_reason_maps_to_error() -> None:
    chunk = GenerateContentResponse.model_validate({"candidates": [{"finish_reason": "NO_IMAGE"}]})
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.finish_reason is FinishReason.ERROR


def test_gemini_unmapped_finish_reason_yields_no_chunk() -> None:
    # OTHER is a real enum member but intentionally unmapped: with no content or
    # tool calls the chunk is suppressed, not emitted as an empty no-op.
    chunk = GenerateContentResponse.model_validate({"candidates": [{"finish_reason": "OTHER"}]})
    assert _gemini_adapter()._to_unified_chunk(chunk) is None


def test_gemini_multi_part_candidate_yields_content_and_tool_call() -> None:
    # One candidate carrying both a text part and a function_call part.
    chunk = GenerateContentResponse.model_validate(
        {"candidates": [{"content": {"parts": [{"text": "hi"}, {"function_call": {"name": "f", "args": {"x": 1}}}]}}]}
    )
    result = _gemini_adapter()._to_unified_chunk(chunk)
    assert result is not None
    assert result.content == "hi"
    assert result.tool_calls is not None
    (call,) = result.tool_calls
    assert call.name == "f"
    # Gemini hands back already-parsed dict args (``function_call.args``); they
    # land on ``ToolCallDelta.args``, not ``args_json`` (the OpenAI/Anthropic
    # string-fragment field). Mirrors pydantic_ai, which passes the dict through.
    assert call.args == {"x": 1}
