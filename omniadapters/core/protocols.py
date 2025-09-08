from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self, runtime_checkable


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
