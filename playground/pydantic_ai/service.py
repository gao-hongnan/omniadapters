from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from omniadapters.pydantic_ai import create_adapter

from .config.settings import PydanticAIDemoConfig


def build_agent_registry(config: PydanticAIDemoConfig) -> dict[str, Agent[None, str]]:
    registry: dict[str, Agent[None, str]] = {}
    for name, agent_cfg in config.agents.items():
        adapter = create_adapter(
            provider_config=agent_cfg.provider_config,
            model_name=agent_cfg.model_name,
        )
        kwargs: dict[str, Any] = {}
        if agent_cfg.system_prompt is not None:
            kwargs["system_prompt"] = agent_cfg.system_prompt
        if agent_cfg.model_settings is not None:
            kwargs["model_settings"] = agent_cfg.model_settings
        registry[name] = adapter.create_agent(**kwargs)
    return registry


async def arun_completion(
    agent: Agent[None, str],
    prompt: str,
    *,
    model_settings: ModelSettings | None = None,
) -> str:
    result = await agent.run(prompt, model_settings=model_settings)
    return result.output
