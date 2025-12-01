from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from instructor import Mode

from omniadapters.core.constants import AZURE_OPENAI_IMPORT_ERROR

try:
    from openai import AsyncAzureOpenAI
    from openai.types import CompletionUsage as OpenAIUsage
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
except ImportError as e:
    raise ImportError(AZURE_OPENAI_IMPORT_ERROR) from e

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import AzureOpenAIProviderConfig, CompletionResponse, StreamChunk, Usage
from omniadapters.core.usage_converter import to_usage


@to_usage.register(OpenAIUsage)
def _(usage: OpenAIUsage) -> Usage:
    cached_input_tokens = None
    thinking_tokens = None

    if usage.prompt_tokens_details:
        cached_input_tokens = usage.prompt_tokens_details.cached_tokens

    if usage.completion_tokens_details:
        thinking_tokens = usage.completion_tokens_details.reasoning_tokens

    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=cached_input_tokens,
        thinking_tokens=thinking_tokens,
    )


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from omniadapters.core.types import MessageParam


class AzureOpenAIAdapter(
    BaseAdapter[
        AzureOpenAIProviderConfig,
        AsyncAzureOpenAI,
        ChatCompletionMessageParam,
        ChatCompletion,
        ChatCompletionChunk,
    ]
):
    @property
    def instructor_mode(self) -> Mode:
        return Mode.TOOLS

    def _create_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(**self.provider_config.get_client_kwargs())

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

        return await self.client.chat.completions.create(
            # messages=formatted_messages,
            # model=self.completion_params.model,
            stream=stream,
            **formatted_params,
        )

    def _to_unified_response(self, response: ChatCompletion) -> CompletionResponse[ChatCompletion]:
        choice = response.choices[0] if response.choices else None

        return CompletionResponse[ChatCompletion](
            content=choice.message.content or "" if choice else "",
            model=response.model,
            usage=to_usage(response.usage) if response.usage else None,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk | None:
        if not chunk.choices:
            return None

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        tool_calls = None
        if delta.tool_calls:
            tool_calls = [tc.model_dump() for tc in delta.tool_calls]

        if not delta.content and not tool_calls and finish_reason is None:
            return None

        return StreamChunk(
            content=delta.content or "",
            model=chunk.model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_chunk=chunk,
        )
