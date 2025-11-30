from __future__ import annotations

import instructor

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message as AnthropicResponse
except ImportError as e:
    raise ImportError(
        "Anthropic provider requires 'anthropic' package. Install with: uv add omniadapters[anthropic]"
    ) from e

from omniadapters.core.models import AnthropicProviderConfig
from omniadapters.structify.adapters.base import BaseAdapter


class AnthropicAdapter(BaseAdapter[AnthropicProviderConfig, AsyncAnthropic, AnthropicResponse]):
    def _create_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(**self.provider_config.get_client_kwargs())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncAnthropic = self.client
        return instructor.from_anthropic(client, mode=self.instructor_config.mode)
