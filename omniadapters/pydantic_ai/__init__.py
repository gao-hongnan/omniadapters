"""Thin wrapper around Pydantic AI for easy agent configuration.

This module provides:
1. `AgentConfig` - A Pydantic model for validated agent configuration
2. `create_agent()` - Factory function to create agents with common settings
3. Re-exports of key Pydantic AI types for convenience

Examples
--------
>>> # Simple usage
>>> from omniadapters.pydantic_ai import create_agent
>>> agent = create_agent("openai:gpt-4o", temperature=0.7)
>>> result = await agent.run("Hello!")

>>> # With config object (supports serialization)
>>> from omniadapters.pydantic_ai import AgentConfig
>>> config = AgentConfig(model="openai:gpt-4o", temperature=0.7, max_tokens=1000)
>>> agent = config.create_agent()

>>> # With structured output
>>> from pydantic import BaseModel
>>> class City(BaseModel):
...     name: str
...     country: str
>>> agent = create_agent("openai:gpt-4o", output_type=City)
>>> result = await agent.run("What's the capital of France?")
>>> print(result.output)  # City(name='Paris', country='France')

"""

from __future__ import annotations

from ..core.constants import PYDANTIC_AI_IMPORT_ERROR

try:
    from pydantic_ai import Agent, ModelRetry, RunContext, Tool
    from pydantic_ai.result import FinalResult, StreamedRunResult
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
except ImportError as e:
    raise ImportError(PYDANTIC_AI_IMPORT_ERROR) from e

from .adapter import PydanticAIAdapter
from .config import AgentConfig, create_agent
from .factory import create_adapter

__all__ = [
    "Agent",
    "AgentConfig",
    "FinalResult",
    "ModelRetry",
    "ModelSettings",
    "PydanticAIAdapter",
    "RequestUsage",
    "RunContext",
    "RunUsage",
    "StreamedRunResult",
    "Tool",
    "UsageLimits",
    "create_adapter",
    "create_agent",
]
