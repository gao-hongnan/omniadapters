"""Unified type definitions for the omniadapters package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, TypeAlias, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import Message, MessageStreamEvent
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from omniadapters.core.models import BaseProviderConfig, CompletionClientParams

MessageParam: TypeAlias = dict[str, Any]
ClientT = TypeVar("ClientT")
ClientResponseT = TypeVar("ClientResponseT", bound="ChatCompletion | Message | GenerateContentResponse")
ProviderConfigT = TypeVar("ProviderConfigT", bound="BaseProviderConfig")
CompletionClientParamsT = TypeVar("CompletionClientParamsT", bound="CompletionClientParams")
StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)
StreamResponseT = TypeVar(
    "StreamResponseT", bound="AsyncIterator[ChatCompletionChunk | MessageStreamEvent | GenerateContentResponse]"
)
