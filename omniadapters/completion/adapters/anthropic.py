"""Documentation: https://docs.claude.com/en/api/messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from instructor import Mode

from ...core.constants import ANTHROPIC_IMPORT_ERROR

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message, RawMessageStreamEvent
    from anthropic.types import MessageParam as AnthropicMessageParam
except ImportError as e:
    raise ImportError(ANTHROPIC_IMPORT_ERROR) from e

from ...core.cost import GENAI_PRICES_PROFILE, UsageExtractionSpec
from ...core.enums import Provider
from ...core.models import AnthropicProviderConfig, CompletionResponse, StreamChunk
from .._map_api_errors import _map_anthropic_errors
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...core.types import MessageParam


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
        content = ""
        if response.content:
            for block in response.content:
                text_val = getattr(block, "text", None)
                if isinstance(text_val, str):
                    content += text_val

        usage = self._extract_usage(response, self._usage_spec, present=response.usage)

        return CompletionResponse[Message](
            content=content,
            model=response.model,
            provider_id=self._usage_spec.provider,
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: RawMessageStreamEvent) -> StreamChunk | None:
        if chunk.type == "content_block_delta":
            delta = getattr(chunk, "delta", None)

            text_val = getattr(delta, "text", None)
            if isinstance(text_val, str):
                return StreamChunk(content=text_val, raw_chunk=chunk)

            partial_json = getattr(delta, "partial_json", None)
            if partial_json is not None:
                return StreamChunk(content="", tool_calls=[{"partial_json": partial_json}], raw_chunk=chunk)

        elif chunk.type == "content_block_start":
            content_block = getattr(chunk, "content_block", None)
            if content_block:
                block_type = getattr(content_block, "type", None)
                if block_type == "tool_use":
                    tool_name = getattr(content_block, "name", None)
                    tool_id = getattr(content_block, "id", None)
                    return StreamChunk(
                        content="",
                        tool_calls=[{"type": "tool_use", "name": tool_name, "id": tool_id}],
                        raw_chunk=chunk,
                    )

        elif chunk.type == "message_stop":
            return StreamChunk(content="", finish_reason="stop", raw_chunk=chunk)
        return None
