from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from .state import AppState


def get_state(request: Request) -> AppState:
    return cast("AppState", request.app.state.app_state)


AppStateDep = Annotated[AppState, Depends(get_state)]
