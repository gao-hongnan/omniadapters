"""Documentation: https://ai.google.dev/gemini-api/docs/text-generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from instructor import Mode
from pydantic_ai.usage import RequestUsage

from ...core.constants import GEMINI_IMPORT_ERROR

try:
    from google import genai
    from google.genai.types import ContentOrDict, GenerateContentConfig, GenerateContentResponse
    from instructor.processing.multimodal import extract_genai_multimodal_content
    from instructor.providers.gemini.utils import (
        convert_to_genai_messages,
        extract_genai_system_message,
        update_genai_kwargs,
    )
except ImportError as e:
    raise ImportError(GEMINI_IMPORT_ERROR) from e

from ...core.models import CompletionResponse, GeminiProviderConfig, StreamChunk
from .._map_api_errors import _map_google_errors
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...core.types import MessageParam


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
        return genai.Client(**self.provider_config.get_client_kwargs())

    def _thanks_instructor(self, messages: list[MessageParam], **kwargs: Any) -> dict[str, Any]:
        """Override because `handle_genai_structured_outputs` is called when response_model is not None."""
        if self.instructor_mode in {Mode.GENAI_STRUCTURED_OUTPUTS, Mode.GENAI_TOOLS}:
            new_kwargs = kwargs.copy()
            new_kwargs["messages"] = messages
            new_kwargs.pop("autodetect_images", False)

            contents = convert_to_genai_messages(messages)  # type: ignore[arg-type]
            contents = extract_genai_multimodal_content(contents, False)
            system_message = extract_genai_system_message(messages)

            generation_config = new_kwargs.get("generation_config", {})
            for param in [
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "seed",
                "presence_penalty",
                "frequency_penalty",
                "max_output_tokens",
            ]:
                if param in new_kwargs:
                    generation_config[param] = new_kwargs.pop(param)

            base_config = {"system_instruction": system_message}
            if generation_config:
                new_kwargs["generation_config"] = generation_config

            final_config = update_genai_kwargs(new_kwargs, base_config)

            final_kwargs = {}
            if "model" in new_kwargs:
                final_kwargs["model"] = new_kwargs["model"]
            else:
                # NOTE: If model wasn't in kwargs (due to exclude=True), get it from completion_params
                final_kwargs["model"] = self.completion_params.model
            final_kwargs["config"] = GenerateContentConfig(**final_config)
            final_kwargs["contents"] = contents

            return final_kwargs

        # NOTE: For other modes, use the parent's implementation
        return super()._thanks_instructor(messages, **kwargs)

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
        formatted_params = self._thanks_instructor(messages, **kwargs)

        with _map_google_errors(model_name=self.completion_params.model):
            if stream:
                return await self.client.aio.models.generate_content_stream(**formatted_params)
            return await self.client.aio.models.generate_content(**formatted_params)

    def _to_unified_response(self, response: GenerateContentResponse) -> CompletionResponse[GenerateContentResponse]:
        content = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text

        model = response.model_version or str(self.completion_params.model)

        # NOTE: genai-prices' google extractor reads camelCase keys
        # (usageMetadata.promptTokenCount, modelVersion, ...), so dump with
        # by_alias=True. The default snake_case dump yields an all-zero usage.
        usage = (
            RequestUsage.extract(
                response.model_dump(mode="python", by_alias=True),
                provider="google",
                provider_url="https://generativelanguage.googleapis.com",
                provider_fallback="google",
                api_flavor="default",
            )
            if response.usage_metadata
            else None
        )

        return CompletionResponse[GenerateContentResponse](
            content=content,
            model=model,
            provider_id="google",
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: GenerateContentResponse) -> StreamChunk | None:
        if not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        content = ""
        tool_calls = None

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text

                if hasattr(part, "function_call"):
                    function_call = part.function_call
                    if function_call is not None:
                        if tool_calls is None:
                            tool_calls = []

                        call_data: dict[str, Any] = {}
                        if hasattr(function_call, "name"):
                            call_data["name"] = function_call.name
                        if hasattr(function_call, "args"):
                            call_data["args"] = function_call.args

                        tool_calls.append(call_data)

        if not content and not tool_calls and not candidate.finish_reason:
            return None

        return StreamChunk(
            content=content,
            finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
            tool_calls=tool_calls,
            raw_chunk=chunk,
        )
