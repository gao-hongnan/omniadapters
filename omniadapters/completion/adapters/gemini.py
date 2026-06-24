"""Documentation: https://ai.google.dev/gemini-api/docs/text-generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, overload

from instructor import Mode

from ...core.constants import GEMINI_IMPORT_ERROR

try:
    from google import genai
    from google.genai.types import ContentOrDict, GenerateContentConfig, GenerateContentResponse
    from google.genai.types import FinishReason as GeminiFinishReason
    from instructor.processing.multimodal import extract_genai_multimodal_content
    from instructor.providers.gemini.utils import (
        convert_to_genai_messages,
        extract_genai_system_message,
        update_genai_kwargs,
    )
except ImportError as e:
    raise ImportError(GEMINI_IMPORT_ERROR) from e

from ...core.cost import GENAI_PRICES_PROFILE, UsageExtractionSpec
from ...core.enums import FinishReason, Provider
from ...core.models import CompletionResponse, GeminiProviderConfig, StreamChunk, ToolCallDelta
from .._map_api_errors import _map_google_errors
from .base import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...core.types import MessageParam


# Gemini's raw `candidate.finish_reason` is an enum; ``str(enum)`` would leak the
# Python class name (``"FinishReason.STOP"``), so map members explicitly. Mirrors
# `pydantic_ai/models/google.py:_FINISH_REASON_MAP`; unmapped members (image-only
# stops, UNSPECIFIED, OTHER) normalize to ``None`` via ``.get``.
_FINISH_REASON_MAP: Final[dict[GeminiFinishReason, FinishReason]] = {
    GeminiFinishReason.STOP: FinishReason.STOP,
    GeminiFinishReason.MAX_TOKENS: FinishReason.LENGTH,
    GeminiFinishReason.SAFETY: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.RECITATION: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.LANGUAGE: FinishReason.ERROR,
    GeminiFinishReason.BLOCKLIST: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.PROHIBITED_CONTENT: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.SPII: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.MALFORMED_FUNCTION_CALL: FinishReason.ERROR,
    GeminiFinishReason.IMAGE_SAFETY: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.UNEXPECTED_TOOL_CALL: FinishReason.ERROR,
    GeminiFinishReason.IMAGE_PROHIBITED_CONTENT: FinishReason.CONTENT_FILTER,
    GeminiFinishReason.NO_IMAGE: FinishReason.ERROR,
}


class GeminiAdapter(
    BaseAdapter[
        GeminiProviderConfig,
        genai.Client,
        ContentOrDict,
        GenerateContentResponse,
        GenerateContentResponse,
    ]
):
    _usage_spec: ClassVar[UsageExtractionSpec] = GENAI_PRICES_PROFILE[Provider.GEMINI]

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

        usage = self._extract_usage(response, self._usage_spec, present=response.usage_metadata)

        return CompletionResponse[GenerateContentResponse](
            content=content,
            model=model,
            provider_id=self._usage_spec.provider,
            usage=usage,
            raw_response=response,
        )

    def _to_unified_chunk(self, chunk: GenerateContentResponse) -> StreamChunk | None:
        if not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        content = ""
        tool_calls: list[ToolCallDelta] | None = None

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    content += part.text

                function_call = part.function_call
                if function_call is not None:
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(ToolCallDelta(name=function_call.name, args=function_call.args))

        raw_reason = candidate.finish_reason
        finish_reason = _FINISH_REASON_MAP.get(raw_reason) if raw_reason else None

        # Suppress no-op chunks: those carrying no text, no tool call, and no
        # *mapped* terminal signal. An unmapped reason (OTHER, UNSPECIFIED) with
        # nothing else is intentionally dropped rather than emitted as an empty
        # chunk — mirroring pydantic_ai, which maps those members to ``None``.
        if not content and not tool_calls and finish_reason is None:
            return None

        return StreamChunk(
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_chunk=chunk,
        )
