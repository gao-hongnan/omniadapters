from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from .config.settings import PydanticAIDemoConfig


AgentRegistry: TypeAlias = "dict[str, Agent[None, str]]"


@dataclass(slots=True, frozen=True)
class AppState:
    config: PydanticAIDemoConfig
    agents: AgentRegistry
