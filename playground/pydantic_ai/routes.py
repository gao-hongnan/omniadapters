from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic_ai.settings import merge_model_settings

from .dependencies import AppStateDep
from .schemas import AgentInfo, CompletionRequest, CompletionResponse
from .service import arun_completion

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents(state: AppStateDep) -> list[AgentInfo]:
    return [
        AgentInfo(name=name, provider=cfg.provider_config.provider, model=cfg.model_name)
        for name, cfg in state.config.agents.items()
    ]


@router.post("/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, state: AppStateDep) -> CompletionResponse:
    if req.agent not in state.agents:
        raise HTTPException(status_code=404, detail=f"agent '{req.agent}' not configured")
    agent = state.agents[req.agent]
    cfg = state.config.agents[req.agent]
    effective = merge_model_settings(cfg.model_settings, req.model_settings)
    output = await arun_completion(agent, req.prompt, model_settings=effective)
    return CompletionResponse(
        agent=req.agent,
        model=cfg.model_name,
        output=output,
        model_settings=effective,
    )
