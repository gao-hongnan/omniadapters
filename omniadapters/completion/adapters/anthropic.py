from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageStreamEvent
from instructor import Mode, handle_response_model

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import AnthropicProviderConfig
from omniadapters.core.types import MessageParam

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionClientParams


class AnthropicAdapter(
    BaseAdapter[
        AnthropicProviderConfig,
        AsyncAnthropic,
        Message,
        AsyncIterator[MessageStreamEvent],
    ]
):
    def __init__(
        self,
        *,
        provider_config: AnthropicProviderConfig,
        completion_params: CompletionClientParams | None = None,
    ) -> None:
        super().__init__(provider_config=provider_config, completion_params=completion_params)

    def _create_client(self) -> AsyncAnthropic:
        config_dict = self.provider_config.model_dump()
        return AsyncAnthropic(**config_dict)

    async def _agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> Message:
        kwargs.pop("stream", None)

        _, formatted = handle_response_model(
            response_model=None,
            mode=Mode.ANTHROPIC_TOOLS,
            messages=messages,
            **kwargs,
        )

        response = await self.client.messages.create(**formatted)
        return cast(Message, response)

    async def _agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> AsyncIterator[MessageStreamEvent]:
        kwargs["stream"] = True

        _, formatted = handle_response_model(
            response_model=None,
            mode=Mode.ANTHROPIC_TOOLS,
            messages=messages,
            **kwargs,
        )

        stream = await self.client.messages.create(**formatted)

        async for event in stream:
            yield event
