# /// script
# dependencies = [
#   "instructor==1.10.0",
#   "pydantic==2.11.7",
#   "pydantic-settings==2.10.1",
# ]
# ///

from __future__ import annotations

from typing import Generic, TypeVar

import instructor
from pydantic import BaseModel, Field

from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)


class InstructorSettings(BaseModel):
    mode: instructor.Mode = Field(default=instructor.Mode.TOOLS)


P = TypeVar("P", OpenAIProviderConfig, AnthropicProviderConfig, GeminiProviderConfig)
C = TypeVar("C", bound=OpenAICompletionClientParams | AnthropicCompletionClientParams | GeminiCompletionClientParams)


class ProviderFamily(BaseModel, Generic[P, C]):
    provider: P
    completion: C
    instructor: InstructorSettings
