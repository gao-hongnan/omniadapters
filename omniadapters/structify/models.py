"""Pydantic models for the structify module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import instructor  # noqa: TC002 - Pydantic needs runtime access to instructor.Mode
from pydantic import BaseModel, ConfigDict

from omniadapters.core.models import Allowable
from omniadapters.core.types import ClientResponseT, StructuredResponseT

if TYPE_CHECKING:
    from omniadapters.structify.hooks import CompletionTrace


class InstructorConfig(Allowable):
    mode: instructor.Mode


class CompletionResult(BaseModel, Generic[StructuredResponseT, ClientResponseT]):
    data: StructuredResponseT
    trace: CompletionTrace[ClientResponseT]

    model_config = ConfigDict(arbitrary_types_allowed=True)
