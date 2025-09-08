from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from google import genai
from google.genai.types import GenerateContentResponse
from instructor import Mode, handle_response_model

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import GeminiProviderConfig
from omniadapters.core.types import MessageParam

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionClientParams


class GeminiAdapter(
    BaseAdapter[
        GeminiProviderConfig,
        genai.Client,
        GenerateContentResponse,
        AsyncIterator[GenerateContentResponse],
    ]
):
    def __init__(
        self,
        *,
        provider_config: GeminiProviderConfig,
        completion_params: CompletionClientParams | None = None,
    ) -> None:
        super().__init__(provider_config=provider_config, completion_params=completion_params)

    def _create_client(self) -> genai.Client:
        config_dict = self.provider_config.model_dump()
        return genai.Client(**config_dict)

    async def _agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> GenerateContentResponse:
        kwargs.pop("stream", None)

        _, formatted = handle_response_model(
            response_model=None,
            mode=Mode.GENAI_TOOLS,
            messages=messages,
            **kwargs,
        )

        model = kwargs.pop("model", self.completion_params.get("model", "gemini-1.5-flash"))

        config = formatted.pop("config", None)
        contents = formatted.pop("contents", [])

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        return response

    async def _agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> AsyncIterator[GenerateContentResponse]:
        kwargs["stream"] = True

        _, formatted = handle_response_model(
            response_model=None,
            mode=Mode.GENAI_TOOLS,
            messages=messages,
            **kwargs,
        )

        model = kwargs.pop("model", self.completion_params.get("model", "gemini-1.5-flash"))

        config = formatted.pop("config", None)
        contents = formatted.pop("contents", [])

        async for response in await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            yield response
