from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import Agent  # NOTE: `Agent` is a heavy import; keep deferred.

    from .config.settings import PydanticAIDemoConfig


AgentRegistry = dict[str, "Agent[None, str]"]


@dataclass(slots=True, frozen=True)
class AppState:
    config: PydanticAIDemoConfig
    agents: AgentRegistry
