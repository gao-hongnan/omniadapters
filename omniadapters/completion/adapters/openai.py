from __future__ import annotations

from typing import Any, AsyncIterator, cast

from instructor import Mode
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import CompletionResponse, CompletionUsage, OpenAIProviderConfig, StreamChunk
from omniadapters.core.types import MessageParam


class OpenAIAdapter(
    BaseAdapter[
        OpenAIProviderConfig,
        AsyncOpenAI,
        ChatCompletion,
        AsyncIterator[ChatCompletionChunk],
    ]
):
    @property
    def instructor_mode(self) -> Mode:
        return Mode.TOOLS

    def _create_client(self) -> AsyncOpenAI:
        config_dict = self.provider_config.model_dump()
        return AsyncOpenAI(**config_dict)

    async def _agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> ChatCompletion:
        kwargs.pop("stream", None)

        params = self._prepare_request(messages, **kwargs)

        coerced_messages = cast(list[ChatCompletionMessageParam], params.pop("messages"))

        response = await self.client.chat.completions.create(
            messages=coerced_messages,
            model=self.completion_params.model,
            extra_body=params,
        )
        return response

    async def _agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs["stream"] = True

        params = self._prepare_request(messages, **kwargs)

        coerced_messages = cast(list[ChatCompletionMessageParam], params.pop("messages"))

        stream = await self.client.chat.completions.create(
            messages=coerced_messages,
            model=self.completion_params.model,
            stream=True,
            extra_body=params,
        )

        async for chunk in stream:
            yield chunk

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
        if not delta.content and chunk.choices[0].finish_reason is None:
            return None

        return StreamChunk(
            content=delta.content or "",
            model=chunk.model,
            finish_reason=chunk.choices[0].finish_reason,
            raw_chunk=chunk,
        )
