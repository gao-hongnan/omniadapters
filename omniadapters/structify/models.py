"""Pydantic models for the structify module."""

from __future__ import annotations

from typing import Generic

import instructor  # noqa: TC002 - Pydantic needs runtime access to instructor.Mode
from pydantic import BaseModel, ConfigDict

from ..core.models import Allowable
from ..core.types import ClientResponseT, StructuredResponseT
from .hooks import CompletionTrace  # noqa: TC001


class InstructorConfig(Allowable):
    mode: instructor.Mode


class CompletionResult(BaseModel, Generic[StructuredResponseT, ClientResponseT]):
    data: StructuredResponseT
    trace: CompletionTrace[ClientResponseT]

    model_config = ConfigDict(arbitrary_types_allowed=True)
