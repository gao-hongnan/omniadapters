from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from anthropic.types import Message, MessageStreamEvent
from anthropic.types import MessageParam as AnthropicMessageParam
from google.genai.types import ContentOrDict, GenerateContentResponse
from instructor.multimodal import Audio, Image
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import BaseModel

if TYPE_CHECKING:
    from omniadapters.core.models import BaseProviderConfig, CompletionClientParams

MessageParam: TypeAlias = dict[str, str | dict[str, Any] | Image | Audio | list[str | dict[str, Any] | Image | Audio]]
ClientT = TypeVar("ClientT")
ClientMessageT = TypeVar(
    "ClientMessageT",
    bound=AnthropicMessageParam | ChatCompletionMessageParam | ContentOrDict,
)
# TODO: consider renaming ClientResponseT to CompletionResponseT
ClientResponseT = TypeVar("ClientResponseT", bound=ChatCompletion | Message | GenerateContentResponse)
ProviderConfigT = TypeVar("ProviderConfigT", bound="BaseProviderConfig")
CompletionClientParamsT = TypeVar("CompletionClientParamsT", bound="CompletionClientParams")
StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)
StreamChunkType: TypeAlias = ChatCompletionChunk | MessageStreamEvent | GenerateContentResponse
StreamChunkT = TypeVar("StreamChunkT", bound=ChatCompletionChunk | MessageStreamEvent | GenerateContentResponse)
