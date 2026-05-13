from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic_ai.settings import ModelSettings


class CompletionRequest(BaseModel):
    agent: str
    prompt: str
    model_settings: ModelSettings | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompletionResponse(BaseModel):
    agent: str
    model: str
    output: str
    model_settings: ModelSettings | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentInfo(BaseModel):
    name: str
    provider: str
    model: str
