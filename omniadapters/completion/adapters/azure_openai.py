from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from instructor import Mode
from pydantic_ai.usage import RequestUsage

from ...core.constants import AZURE_OPENAI_IMPORT_ERROR

try:
    from openai import AsyncAzureOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
except ImportError as e:
    raise ImportError(AZURE_OPENAI_IMPORT_ERROR) from e

from ...core.models import AzureOpenAIProviderConfig, CompletionResponse, StreamChunk
from .._map_api_errors import _map_azure_openai_errors
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...core.types import MessageParam


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

        with _map_azure_openai_errors(model_name=self.completion_params.model):
            return await self.client.chat.completions.create(stream=stream, **formatted_params)

    def _to_unified_response(self, response: ChatCompletion) -> CompletionResponse[ChatCompletion]:
        choice = response.choices[0] if response.choices else None

        # NOTE: Azure responses carry the OpenAI ChatCompletion usage shape, so
        # the OpenAI 'chat' extractor parses them. Prefer the configured Azure
        # endpoint for provider resolution, falling back to the OpenAI URL and
        # the 'openai' fallback id so genai-prices still finds the model.
        azure_endpoint = getattr(self.provider_config, "azure_endpoint", None)
        provider_url = (
            azure_endpoint if isinstance(azure_endpoint, str) and azure_endpoint else "https://api.openai.com"
        )

        usage = (
            RequestUsage.extract(
                response.model_dump(mode="python"),
                provider="azure",
                provider_url=provider_url,
                provider_fallback="openai",
                api_flavor="chat",
            )
            if response.usage
            else None
        )

        return CompletionResponse[ChatCompletion](
            content=choice.message.content or "" if choice else "",
            model=response.model,
            provider_id="azure",
            usage=usage,
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
