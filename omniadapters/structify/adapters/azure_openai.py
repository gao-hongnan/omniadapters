from __future__ import annotations

import instructor

from ...core.constants import AZURE_OPENAI_IMPORT_ERROR

try:
    from openai import AsyncAzureOpenAI
    from openai.types.chat import ChatCompletion
except ImportError as e:
    raise ImportError(AZURE_OPENAI_IMPORT_ERROR) from e

from ...core.models import AzureOpenAIProviderConfig
from .base import BaseAdapter


class AzureOpenAIAdapter(BaseAdapter[AzureOpenAIProviderConfig, AsyncAzureOpenAI, ChatCompletion]):
    def _create_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(**self.provider_config.model_dump())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncAzureOpenAI = self.client
        return instructor.from_openai(client, mode=self.instructor_config.mode)
