from __future__ import annotations

from pydantic_ai import Agent

from omniadapters.pydantic_ai import create_adapter

from .config.settings import PydanticAIDemoConfig


def build_agent_registry(config: PydanticAIDemoConfig) -> dict[str, Agent[None, str]]:
    registry: dict[str, Agent[None, str]] = {}
    for name, agent_cfg in config.agents.items():
        adapter = create_adapter(
            provider_config=agent_cfg.provider_config,
            model_name=agent_cfg.model_name,
        )
        kwargs: dict[str, str] = {}
        if agent_cfg.system_prompt is not None:
            kwargs["system_prompt"] = agent_cfg.system_prompt
        registry[name] = adapter.create_agent(**kwargs)
    return registry


async def arun_completion(agent: Agent[None, str], prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output
