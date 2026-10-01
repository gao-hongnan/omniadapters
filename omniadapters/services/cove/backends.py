"""The courtroom's LLM backend: a pydantic-ai agent per role, built in code or from YAML and the environment.

Kept out of the package root, so the engine and procedures never import a provider SDK.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Annotated, assert_never

from pydantic import BaseModel, ConfigDict, Field, with_config
from pydantic_ai import Agent, AgentRetries, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.settings import ModelSettings

from omniadapters.core.models import ProviderConfig  # noqa: TC001 - pydantic resolves it at runtime
from omniadapters.pydantic_ai import build_model

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.output import OutputSpec

    from .prompts import Messages


class OutputMode(StrEnum):
    """How a role's model is made to answer in a step's response model: pydantic-ai's output modes.

    - ``AUTO``: whatever the model's profile prefers, which is a tool call for most models.
    - ``TOOL``: the model calls a tool whose arguments are the response.
    - ``NATIVE``: the provider's own structured output holds the response to the JSON schema.
    - ``PROMPTED``: the JSON schema goes into the instructions and the response is parsed from text,
      for models that support neither tools nor structured output.
    """

    AUTO = auto()
    TOOL = auto()
    NATIVE = auto()
    PROMPTED = auto()


class PydanticAIStructuredLLM:
    """A :class:`~omniadapters.services.cove.llm.StructuredLLM` that runs a pydantic-ai agent once per call.

    The system message is added to the agent's own instructions, and the user message is the prompt.
    pydantic-ai validates the response as the step's ``response_model``; when validation fails, it
    shows the model the errors and lets it try again, as often as the agent's output retries allow.
    An agent with tools may use them before it answers.

    Build the agent with ``omniadapters.pydantic_ai.create_agent``, or let :func:`open_role` build it
    from a :class:`RoleConfig`. The response model replaces the agent's output type on every call, so
    the agent must not have output validators.

    Parameters
    ----------
    agent
        The model behind the role. Pass the same object to every step the role plays.
    output_mode
        How the model is made to answer in the response model.

    """

    __slots__ = ("_agent", "_output_mode")

    def __init__(self, agent: AbstractAgent[None, object], *, output_mode: OutputMode = OutputMode.AUTO) -> None:
        self._agent = agent
        self._output_mode = output_mode

    @property
    def agent(self) -> AbstractAgent[None, object]:
        """The agent behind the role, for example to ``override`` its model in a test."""
        return self._agent

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Run the agent once, with ``response_model`` as the run's output type.

        Raises
        ------
        pydantic_ai.exceptions.UnexpectedModelBehavior
            If no response validates within the agent's output retries.

        """
        result = await self._agent.run(
            messages.user,
            output_type=_output_spec(response_model, self._output_mode),
            instructions=messages.system or None,
            # Otherwise pydantic-ai searches this frame for a variable naming the agent, on every call, in vain.
            infer_name=False,
        )
        return result.output


def _output_spec[ResponseT: BaseModel](response_model: type[ResponseT], mode: OutputMode) -> OutputSpec[ResponseT]:
    match mode:
        case OutputMode.AUTO:
            return response_model
        case OutputMode.TOOL:
            return ToolOutput(response_model)
        case OutputMode.NATIVE:
            return NativeOutput(response_model)
        case OutputMode.PROMPTED:
            return PromptedOutput(response_model)
        case _:
            assert_never(mode)


@with_config(ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _StrictModelSettings(ModelSettings, total=False):
    """pydantic-ai's ``ModelSettings``, validated so that a misspelt key fails instead of being ignored.

    ``arbitrary_types_allowed`` stays scoped to the settings, where it admits ``timeout``'s
    ``httpx.Timeout``; config gives the timeout in seconds.
    """


class RoleConfig(BaseModel):
    """The model behind one role, exactly as it appears in YAML or the environment.

    Unknown keys fail validation instead of being ignored, model settings included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_config: ProviderConfig
    model_name: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The model, by name. A bare name is served by the provider's default API; a pydantic-ai "
                "API prefix such as `openai-responses:` picks another API of the same provider."
            ),
        ),
    ]
    model_settings: Annotated[
        _StrictModelSettings,
        Field(
            default_factory=_StrictModelSettings,
            description="pydantic-ai model settings: `temperature`, `max_tokens`, `thinking`, `timeout` in seconds.",
        ),
    ]
    output_mode: Annotated[
        OutputMode, Field(description="How the model is made to answer in each step's response model.")
    ] = OutputMode.AUTO
    output_retries: Annotated[
        int,
        Field(
            ge=0,
            le=10,
            description=(
                "Retries after a response fails validation; each shows the model the errors and costs one more request."
            ),
        ),
    ] = 1


@asynccontextmanager
async def open_role(config: RoleConfig, *, name: str | None = None) -> AsyncGenerator[PydanticAIStructuredLLM]:
    """Build one agent for a role, and close its HTTP client on exit, even on error.

    Open each role once and reuse it for every call the role makes. A witness fan-out then
    shares one connection pool instead of building a client per call.

    Parameters
    ----------
    config
        The role's provider, model, model settings, output mode and output retries.
    name
        Names the agent in pydantic-ai's traces, for example after the role it plays.

    Yields
    ------
    PydanticAIStructuredLLM
        The role's model, ready for :func:`~omniadapters.services.cove.llm.llm_step`.

    """
    agent = Agent(
        build_model(provider_config=config.provider_config, model_name=config.model_name),
        name=name,
        model_settings=config.model_settings,
        retries=AgentRetries(output=config.output_retries),
    )
    async with agent:
        yield PydanticAIStructuredLLM(agent, output_mode=config.output_mode)
