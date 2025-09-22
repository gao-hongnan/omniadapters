from __future__ import annotations

from typing import Any, AsyncIterator, Literal, overload

from instructor import Mode
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import AzureOpenAIProviderConfig, CompletionResponse, CompletionUsage, StreamChunk
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
        config_dict = self.provider_config.model_dump()
        return AsyncAzureOpenAI(**config_dict)

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

        response = await self.client.chat.completions.create(
            # messages=formatted_messages,
            # model=self.completion_params.model,
            stream=stream,
            **formatted_params,
        )
        return response

    def _to_unified_response(self, response: ChatCompletion) -> CompletionResponse[ChatCompletion]:
        choice = response.choices[0] if response.choices else None
        return CompletionResponse[ChatCompletion](
            content=choice.message.content or "" if choice else "",
            model=response.model,
            usage=CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            if response.usage
            else None,
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
