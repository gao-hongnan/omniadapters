from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from .config.settings import PydanticAIDemoConfig
from .state import AgentRegistry


def build_agent_registry(config: PydanticAIDemoConfig) -> AgentRegistry:
    return {name: cfg.build_agent() for name, cfg in config.agents.items()}


async def arun_completion(
    agent: Agent[None, str],
    prompt: str,
    *,
    model_settings: ModelSettings | None = None,
) -> str:
    result = await agent.run(prompt, model_settings=model_settings)
    return result.output
