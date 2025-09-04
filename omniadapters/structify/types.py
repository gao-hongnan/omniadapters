"""Type definitions and aliases for the structify module."""

from __future__ import annotations

from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicResponse
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion

    from omniadapters.structify.models import BaseProviderConfig, CompletionClientParams

StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)
BaseProviderConfigT = TypeVar("BaseProviderConfigT", bound="BaseProviderConfig")
ClientT = TypeVar("ClientT")
CompletionClientParamsT = TypeVar("CompletionClientParamsT", bound="CompletionClientParams")
ClientResponseT = TypeVar(
    "ClientResponseT",
    bound="ChatCompletion | AnthropicResponse | GenerateContentResponse",
)

MessageParam: TypeAlias = dict[str, Any]


@runtime_checkable
class AsyncCloseable(Protocol):
    """Protocol for clients with async close method."""

    async def close(self) -> None: ...


@runtime_checkable
class AsyncContextManager(Protocol):
    """Protocol for async context manager support."""

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
