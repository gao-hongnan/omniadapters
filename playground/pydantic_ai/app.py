from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import get_settings
from .constants import APP_TITLE, ENV_FILE_VAR, YAML_FILE_VAR
from .routes import router
from .service import build_agent_registry
from .state import AppState


def create_app(*, env_file: str | None = None, yaml_file: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        settings = get_settings(env_file=env_file, yaml_file=yaml_file)
        app.state.app_state = AppState(
            config=settings.pydantic_ai_demo,
            agents=build_agent_registry(settings.pydantic_ai_demo),
        )
        yield

    app = FastAPI(title=APP_TITLE, lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app(
    env_file=os.environ.get(ENV_FILE_VAR),
    yaml_file=os.environ.get(YAML_FILE_VAR),
)
