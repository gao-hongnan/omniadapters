from __future__ import annotations

import instructor

from ...core.constants import OPENAI_IMPORT_ERROR

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
except ImportError as e:
    raise ImportError(OPENAI_IMPORT_ERROR) from e

from ...core.models import OpenAIProviderConfig
from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter[OpenAIProviderConfig, AsyncOpenAI, ChatCompletion]):
    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self.provider_config.get_client_kwargs())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncOpenAI = self.client
        return instructor.from_openai(client, mode=self.instructor_config.mode)
