from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings, merge_model_settings

from .config.settings import get_settings
from .service import arun_completion, build_agent_registry


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings(
        env_file=os.environ.get("PYDANTIC_AI_DEMO_ENV_FILE"),
        yaml_file=os.environ.get("PYDANTIC_AI_DEMO_YAML_FILE"),
    )
    app.state.config = settings.pydantic_ai_demo
    app.state.agents = build_agent_registry(settings.pydantic_ai_demo)
    yield


app = FastAPI(title="pydantic-ai demo", lifespan=lifespan)


def _get_agent(name: str, request: Request) -> Agent[None, str]:
    agents: dict[str, Agent[None, str]] = request.app.state.agents
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"agent '{name}' not configured")
    return agents[name]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents", response_model=list[AgentInfo])
async def list_agents(request: Request) -> list[AgentInfo]:
    config = request.app.state.config
    return [
        AgentInfo(name=name, provider=cfg.provider_config.provider, model=cfg.model_name)
        for name, cfg in config.agents.items()
    ]


@app.post("/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, request: Request) -> CompletionResponse:
    agent = _get_agent(req.agent, request)
    cfg_agent = request.app.state.config.agents[req.agent]
    effective_settings = merge_model_settings(cfg_agent.model_settings, req.model_settings)
    output = await arun_completion(agent, req.prompt, model_settings=effective_settings)
    return CompletionResponse(
        agent=req.agent,
        model=cfg_agent.model_name,
        output=output,
        model_settings=effective_settings,
    )
