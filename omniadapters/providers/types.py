from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, TypeAlias, TypeVar

if TYPE_CHECKING:
    from anthropic.types import Message, MessageStreamEvent
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from omniadapters.core.models import BaseProviderConfig

MessageParam: TypeAlias = dict[str, Any]
ProviderConfigT = TypeVar("ProviderConfigT", bound="BaseProviderConfig")
ClientT = TypeVar("ClientT")
ClientResponseT = TypeVar("ClientResponseT", bound="ChatCompletion | Message | GenerateContentResponse")
StreamResponseT = TypeVar(
    "StreamResponseT", bound="AsyncIterator[ChatCompletionChunk | MessageStreamEvent | GenerateContentResponse]"
)
