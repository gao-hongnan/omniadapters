"""Shared base for OpenAI-wire-compatible adapters (OpenAI + Azure OpenAI).

Both providers speak the identical Chat Completions schema, so the stream
parsing, finish-reason normalization, response mapping, and instructor mode all
live here exactly once. Concrete adapters supply only what genuinely differs:
the client class, its error-mapping context manager, the usage spec, and (for
Azure) the dynamic provider URL.

The finish-reason table mirrors ``pydantic_ai/models/openai.py``'s
``_CHAT_FINISH_REASON_MAP``; ``tool_use``/``function_call`` both normalize to
:attr:`FinishReason.TOOL_CALL`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeVar, overload

from instructor import Mode

from ...core.constants import OPENAI_IMPORT_ERROR

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
except ImportError as e:
    raise ImportError(OPENAI_IMPORT_ERROR) from e

from ...core.enums import FinishReason
from ...core.models import CompletionResponse, StreamChunk, ToolCallDelta
from ...core.types import ProviderConfigT
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager

    from ...core.cost import UsageExtractionSpec
    from ...core.types import MessageParam

OpenAIClientT = TypeVar("OpenAIClientT", bound=AsyncOpenAI)

# Raw `choice.finish_reason` literal emitted by the Chat Completions API.
_OpenAIChatFinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]

_FINISH_REASON_MAP: Final[dict[_OpenAIChatFinishReason, FinishReason]] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "content_filter": FinishReason.CONTENT_FILTER,
    "function_call": FinishReason.TOOL_CALL,
}


class OpenAICompatibleAdapter(
    BaseAdapter[ProviderConfigT, OpenAIClientT, ChatCompletionMessageParam, ChatCompletion, ChatCompletionChunk],
):
    _usage_spec: ClassVar[UsageExtractionSpec]

    @property
    def instructor_mode(self) -> Mode:
        return Mode.TOOLS

    @abstractmethod
    def _create_client(self) -> OpenAIClientT: ...

    @abstractmethod
    def _map_errors(self, *, model_name: str) -> AbstractContextManager[None]:
        """Return the vendor's exception-translation context manager."""

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ChatCompletion: ...

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        formatted_params = self._thanks_instructor(messages, **kwargs)

        with self._map_errors(model_name=self.completion_params.model):
            return await self.client.chat.completions.create(stream=stream, **formatted_params)

    def _to_unified_response(self, response: ChatCompletion) -> CompletionResponse[ChatCompletion]:
        choice = response.choices[0] if response.choices else None

        usage = self._extract_usage(response, self._usage_spec, present=response.usage)

        return CompletionResponse[ChatCompletion](
            content=choice.message.content or "" if choice else "",
            model=response.model,
            provider_id=self._usage_spec.provider,
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk | None:
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta
        raw_finish_reason = choice.finish_reason
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason) if raw_finish_reason else None

        tool_calls: list[ToolCallDelta] | None = None
        if delta.tool_calls:
            tool_calls = [
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    name=tc.function.name if tc.function else None,
                    args_json=tc.function.arguments if tc.function else None,
                )
                for tc in delta.tool_calls
            ]

        if not delta.content and not tool_calls and finish_reason is None:
            return None

        return StreamChunk(
            content=delta.content or "",
            model=chunk.model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_chunk=chunk,
        )
