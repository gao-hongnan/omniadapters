"""Documentation: https://ai.google.dev/gemini-api/docs/text-generation"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Literal, overload

from google import genai
from google.genai.types import ContentOrDict, GenerateContentConfig, GenerateContentResponse
from instructor import Mode
from instructor.multimodal import extract_genai_multimodal_content
from instructor.utils import convert_to_genai_messages, extract_genai_system_message, update_genai_kwargs

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

    def _thanks_instructor(self, messages: list[MessageParam], **kwargs: Any) -> dict[str, Any]:
        """Override cause `handle_genai_structured_outputs` is called when `response_model` is not `None` but we are using `None`."""

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

        if stream:
            return await self.client.aio.models.generate_content_stream(
                # model=model,
                # contents=formatted_messages,
                # config=cast(GenerateContentConfigDict, kwargs) if kwargs else None,
                **formatted_params,
            )
        else:
            response = await self.client.aio.models.generate_content(
                # model=model,
                # contents=formatted_messages,
                # config=cast(GenerateContentConfigDict, kwargs) if kwargs else None,
                **formatted_params,
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

    async def aclose(self) -> None:
        """Can't seem to find gemini's closure method so we custom close the adapter and cleanup resources."""
        try:
            if self._client:
                self._client._api_client._httpx_client.close()
                await self._client._api_client._async_httpx_client.aclose()
        except Exception as e:
            logging.warning(f"Error closing Gemini client: {e}")
        finally:
            await super().aclose()
