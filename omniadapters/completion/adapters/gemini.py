from __future__ import annotations

from typing import Any, AsyncIterator, Literal, cast, overload

from google import genai
from google.genai.types import ContentOrDict, GenerateContentConfigDict, GenerateContentResponse
from instructor import Mode

from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.core.models import CompletionResponse, CompletionUsage, GeminiProviderConfig, StreamChunk
from omniadapters.core.types import MessageParam


class GeminiAdapter(
    BaseAdapter[
        GeminiProviderConfig,
        genai.Client,
        ContentOrDict,
        GenerateContentResponse,
        GenerateContentResponse,
    ]
):
    @property
    def instructor_mode(self) -> Mode:
        return Mode.GENAI_STRUCTURED_OUTPUTS

    def _create_client(self) -> genai.Client:
        config_dict = self.provider_config.model_dump()
        return genai.Client(**config_dict)

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> GenerateContentResponse: ...

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[GenerateContentResponse]: ...

    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateContentResponse | AsyncIterator[GenerateContentResponse]:
        formatted_messages = self._format_messages(messages, **kwargs)

        model = self.completion_params.model

        if stream:
            return await self.client.aio.models.generate_content_stream(
                model=model,
                contents=formatted_messages,
                config=cast(GenerateContentConfigDict, kwargs) if kwargs else None,
            )
        else:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=formatted_messages,
                config=cast(GenerateContentConfigDict, kwargs) if kwargs else None,
            )
            return response

    def _to_unified_response(self, response: GenerateContentResponse) -> CompletionResponse[GenerateContentResponse]:
        content = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text

        model = response.model_version or str(self.completion_params.model)

        usage = None
        if response.usage_metadata:
            usage = CompletionUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        return CompletionResponse[GenerateContentResponse](
            content=content,
            model=model,
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: GenerateContentResponse) -> StreamChunk | None:
        if not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        content = ""

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text

        if not content and not candidate.finish_reason:
            return None

        return StreamChunk(
            content=content,
            finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
            raw_chunk=chunk,
        )
