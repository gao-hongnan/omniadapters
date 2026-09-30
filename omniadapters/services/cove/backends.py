"""Concrete LLMs for the courtroom: omniadapters structify adapters (config-driven) and pydantic-ai agents.

Kept out of the package root, so the engine and procedures never import a provider SDK.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, model_validator

from omniadapters.core.models import (  # noqa: TC001 - pydantic resolves these at runtime
    CompletionClientParams,
    ProviderConfig,
)
from omniadapters.structify import create_adapter
from omniadapters.structify.models import InstructorConfig  # noqa: TC001 - pydantic resolves this at runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai import Agent

    from .llm import StructuredLLM
    from .prompts import Messages


class RoleConfig(BaseModel):
    """The LLM behind one role, exactly as it appears in YAML or the environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_config: ProviderConfig
    completion_params: CompletionClientParams
    instructor_config: InstructorConfig

    @model_validator(mode="after")
    def _providers_agree(self) -> Self:
        if self.provider_config.provider != self.completion_params.provider:
            msg = (
                f"provider mismatch: provider_config is {self.provider_config.provider!r} "
                f"but completion_params is {self.completion_params.provider!r}"
            )
            raise ValueError(msg)
        return self


@asynccontextmanager
async def open_role(config: RoleConfig) -> AsyncGenerator[StructuredLLM]:
    """Open one structify adapter for a role, and close its HTTP client on exit, even on error.

    Open each role once and reuse it for every call the role makes. A witness fan-out then
    shares one connection pool instead of building a client per call.
    """
    adapter = create_adapter(
        provider_config=config.provider_config,
        completion_params=config.completion_params,
        instructor_config=config.instructor_config,
    )
    try:
        yield adapter
    finally:
        await adapter.aclose()


class PydanticAIStructuredLLM:
    """A :class:`~omniadapters.services.cove.llm.StructuredLLM` over a pydantic-ai ``Agent``.

    System messages become the run's instructions and user messages its prompt. Build the agent
    with ``omniadapters.pydantic_ai.create_agent(provider_config=..., model_name=...)``.
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: Agent[None, str]) -> None:
        self._agent = agent

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Run the agent once with ``response_model`` as its output type."""
        instructions = _text_of(messages, role="system")
        result = await self._agent.run(
            _text_of(messages, role="user"),
            output_type=response_model,
            instructions=instructions or None,
        )
        return result.output


def _text_of(messages: Messages, *, role: str) -> str:
    parts: list[str] = []
    for message in messages:
        if message["role"] != role:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            msg = f"PydanticAIStructuredLLM supports text-only {role} messages, got {type(content).__name__}"
            raise TypeError(msg)
        parts.append(content)
    return "\n\n".join(parts)
