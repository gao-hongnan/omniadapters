from __future__ import annotations

import instructor

_OPENAI_IMPORT_ERROR = "OpenAI provider requires 'openai' package. Install with: uv add omniadapters[openai]"

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
except ImportError as e:
    raise ImportError(_OPENAI_IMPORT_ERROR) from e

from omniadapters.core.models import OpenAIProviderConfig
from omniadapters.structify.adapters.base import BaseAdapter


class OpenAIAdapter(BaseAdapter[OpenAIProviderConfig, AsyncOpenAI, ChatCompletion]):
    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self.provider_config.get_client_kwargs())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncOpenAI = self.client
        return instructor.from_openai(client, mode=self.instructor_config.mode)
