from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType


@runtime_checkable  # NOTE: for `isinstance` check
class AsyncCloseable(Protocol):
    """Protocol for clients with an async `close` method."""

    async def close(self) -> None: ...


@runtime_checkable
class AsyncACloseable(Protocol):
    """Protocol for clients with an async `aclose` method."""

    async def aclose(self) -> None: ...


@runtime_checkable
class GeminiAClose(Protocol):
    """Protocol for clients exposing an `aio` attribute with `aclose`.

    Example: `google.genai.Client` where `client.aio` has `aclose`.
    """

    @property
    def aio(self) -> AsyncACloseable: ...


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
