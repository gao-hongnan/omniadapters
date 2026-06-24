"""Documentation: https://docs.claude.com/en/api/messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, assert_never, overload

from instructor import Mode

from ...core.constants import ANTHROPIC_IMPORT_ERROR

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        InputJSONDelta,
        Message,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
        RawMessageStreamEvent,
        TextBlock,
        TextDelta,
        ToolUseBlock,
    )
    from anthropic.types import MessageParam as AnthropicMessageParam
except ImportError as e:
    raise ImportError(ANTHROPIC_IMPORT_ERROR) from e

from ...core.cost import GENAI_PRICES_PROFILE, UsageExtractionSpec
from ...core.enums import FinishReason, Provider
from ...core.models import AnthropicProviderConfig, CompletionResponse, StreamChunk, ToolCallDelta
from .._map_api_errors import _map_anthropic_errors
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types import StopReason

    from ...core.types import MessageParam


# Anthropic delivers the real stop reason on the `message_delta` event, not on
# `message_stop`. Mirrors `pydantic_ai/models/anthropic.py:_FINISH_REASON_MAP`.
_FINISH_REASON_MAP: Final[dict[StopReason, FinishReason]] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "pause_turn": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALL,
    "refusal": FinishReason.CONTENT_FILTER,
}


class AnthropicAdapter(
    BaseAdapter[
        AnthropicProviderConfig,
        AsyncAnthropic,
        AnthropicMessageParam,
        Message,
        RawMessageStreamEvent,
    ]
):
    _usage_spec: ClassVar[UsageExtractionSpec] = GENAI_PRICES_PROFILE[Provider.ANTHROPIC]

    @property
    def instructor_mode(self) -> Mode:
        return Mode.ANTHROPIC_TOOLS

    def _create_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(**self.provider_config.get_client_kwargs())

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> Message: ...

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[RawMessageStreamEvent]: ...

    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Message | AsyncIterator[RawMessageStreamEvent]:
        formatted_params = self._thanks_instructor(messages, **kwargs)
        # NOTE: least overload needed requires model and max_tokens!
        with _map_anthropic_errors(model_name=self.completion_params.model):
            return await self.client.messages.create(stream=stream, **formatted_params)

    def _to_unified_response(self, response: Message) -> CompletionResponse[Message]:
        content = "".join(block.text for block in response.content if isinstance(block, TextBlock))

        usage = self._extract_usage(response, self._usage_spec, present=response.usage)

        return CompletionResponse[Message](
            content=content,
            model=response.model,
            provider_id=self._usage_spec.provider,
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: RawMessageStreamEvent) -> StreamChunk | None:
        if isinstance(chunk, RawContentBlockDeltaEvent):
            return self._content_block_delta_to_chunk(chunk)
        if isinstance(chunk, RawContentBlockStartEvent):
            return self._content_block_start_to_chunk(chunk)
        if isinstance(chunk, RawMessageDeltaEvent):
            raw_reason = chunk.delta.stop_reason
            if raw_reason is None:
                return None
            return StreamChunk(content="", finish_reason=_FINISH_REASON_MAP.get(raw_reason), raw_chunk=chunk)
        if isinstance(chunk, (RawMessageStartEvent, RawContentBlockStopEvent, RawMessageStopEvent)):
            return None
        # Exhaustive: a new `RawMessageStreamEvent` member fails type-checking here.
        assert_never(chunk)

    @staticmethod
    def _content_block_delta_to_chunk(chunk: RawContentBlockDeltaEvent) -> StreamChunk | None:
        delta = chunk.delta
        if isinstance(delta, TextDelta):
            return StreamChunk(content=delta.text, raw_chunk=chunk)
        if isinstance(delta, InputJSONDelta):
            return StreamChunk(content="", tool_calls=[ToolCallDelta(args_json=delta.partial_json)], raw_chunk=chunk)
        return None

    @staticmethod
    def _content_block_start_to_chunk(chunk: RawContentBlockStartEvent) -> StreamChunk | None:
        block = chunk.content_block
        if isinstance(block, ToolUseBlock):
            return StreamChunk(content="", tool_calls=[ToolCallDelta(id=block.id, name=block.name)], raw_chunk=chunk)
        return None
