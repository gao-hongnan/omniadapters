from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic_ai.settings import ModelSettings


class CompletionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: str
    prompt: str
    model_settings: ModelSettings | None = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: str
    model: str
    output: str
    model_settings: ModelSettings | None = None


class AgentInfo(BaseModel):
    name: str
    provider: str
    model: str
