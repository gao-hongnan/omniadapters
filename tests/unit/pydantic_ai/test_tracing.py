from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from omniadapters.pydantic_ai.tracing import PydanticAITrace, make_trace_recorder


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


@pytest.mark.unit
class TestMakeTraceRecorder:
    def test_returns_handler_and_empty_trace(self) -> None:
        handler, trace = make_trace_recorder()
        assert callable(handler)
        assert isinstance(trace, PydanticAITrace)
        assert trace.events == []

    @pytest.mark.asyncio
    async def test_handler_collects_events_in_order(self) -> None:
        handler, trace = make_trace_recorder()
        sentinel_events: list[Any] = [object(), object(), object()]

        await handler(None, _aiter(sentinel_events))  # type: ignore[arg-type]

        assert list(trace.events) == sentinel_events

    @pytest.mark.asyncio
    async def test_each_recorder_has_independent_trace(self) -> None:
        h1, t1 = make_trace_recorder()
        h2, t2 = make_trace_recorder()
        a, b, c = object(), object(), object()

        await h1(None, _aiter([a]))  # type: ignore[arg-type]
        await h2(None, _aiter([b, c]))  # type: ignore[arg-type]

        assert list(t1.events) == [a]
        assert list(t2.events) == [b, c]

    @pytest.mark.asyncio
    async def test_handler_handles_empty_stream(self) -> None:
        handler, trace = make_trace_recorder()
        await handler(None, _aiter([]))  # type: ignore[arg-type]
        assert trace.events == []

    @pytest.mark.asyncio
    async def test_handler_propagates_iterator_exception(self) -> None:
        handler, trace = make_trace_recorder()
        first = object()

        async def _broken() -> AsyncIterator[Any]:
            yield first
            raise RuntimeError("stream failed")

        with pytest.raises(RuntimeError, match="stream failed"):
            await handler(None, _broken())  # type: ignore[arg-type]

        # Events emitted before the failure are still recorded.
        assert list(trace.events) == [first]
