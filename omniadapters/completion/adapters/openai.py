from __future__ import annotations

from typing import Any, AsyncIterator, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import OpenAIProviderConfig
from omniadapters.core.types import MessageParam


class OpenAIAdapter(
    BaseAdapter[
        OpenAIProviderConfig,
        AsyncOpenAI,
        ChatCompletion,
        AsyncIterator[ChatCompletionChunk],
    ]
):
    def _create_client(self) -> AsyncOpenAI:
        config_dict = self.provider_config.model_dump()
        return AsyncOpenAI(**config_dict)

    async def _agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> ChatCompletion:
        kwargs.pop("stream", None)

        openai_messages = cast(list[ChatCompletionMessageParam], messages)
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            **kwargs,
        )
        return cast(ChatCompletion, response)

    async def _agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs["stream"] = True

        openai_messages = cast(list[ChatCompletionMessageParam], messages)
        stream = await self.client.chat.completions.create(
            messages=openai_messages,
            **kwargs,
        )

        async for chunk in stream:
            yield chunk
