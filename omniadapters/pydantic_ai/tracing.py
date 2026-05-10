"""Lightweight tracing helper for ``pydantic_ai.Agent``.

Pydantic-ai exposes an ``event_stream_handler`` per call on
``Agent.run`` / ``Agent.run_stream``. This module supplies a tiny recorder
that collects every emitted event into an in-memory list, so callers who
want a structify-style trace can opt in without us re-implementing
instructor's five-hook lifecycle.

Usage::

    from omniadapters.pydantic_ai import make_trace_recorder

    handler, trace = make_trace_recorder()
    result = await agent.run("hello", event_stream_handler=handler)
    print(trace.events)          # list[AgentStreamEvent]
    print(result.all_messages()) # canonical model messages
    print(result.usage())        # token usage

The recorder is intentionally minimal. For richer instrumentation
(spans, metrics, log forwarding) configure pydantic-ai's
``instrument=`` / Logfire integration directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Awaitable, Callable

    from pydantic_ai._run_context import RunContext
    from pydantic_ai.messages import AgentStreamEvent

__all__ = ["PydanticAITrace", "make_trace_recorder"]


@dataclass
class PydanticAITrace:
    """Captured stream events from an ``Agent.run`` invocation."""

    events: list[AgentStreamEvent] = field(default_factory=list)


def make_trace_recorder() -> tuple[
    Callable[[RunContext[object], AsyncIterable[AgentStreamEvent]], Awaitable[None]],
    PydanticAITrace,
]:
    """Return ``(handler, trace)`` for use as ``event_stream_handler=``.

    The handler drains the supplied async iterable and appends each event
    to ``trace.events``. The trace is shared by reference, so reads after
    the run completes see the full event log.
    """
    trace = PydanticAITrace()

    async def handler(
        _ctx: RunContext[object],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        async for event in events:
            trace.events.append(event)

    return handler, trace
