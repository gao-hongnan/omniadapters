from __future__ import annotations

import instructor
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from omniadapters.core.models import AzureOpenAIProviderConfig
from omniadapters.structify.adapters.base import BaseAdapter


class AzureOpenAIAdapter(BaseAdapter[AzureOpenAIProviderConfig, AsyncAzureOpenAI, ChatCompletion]):
    def _create_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(**self.provider_config.model_dump())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncAzureOpenAI = self.client
        return instructor.from_openai(client, mode=self.instructor_config.mode)
