"""Pydantic models for the structify module."""

from __future__ import annotations

from typing import Generic

import instructor
from pydantic import BaseModel, ConfigDict

from omniadapters.core.models import Allowable
from omniadapters.core.types import ClientResponseT, StructuredResponseT
from omniadapters.structify.hooks import CompletionTrace


class InstructorConfig(Allowable):
    mode: instructor.Mode


class CompletionResult(BaseModel, Generic[StructuredResponseT, ClientResponseT]):
    data: StructuredResponseT
    trace: CompletionTrace[ClientResponseT]

    model_config = ConfigDict(arbitrary_types_allowed=True)
