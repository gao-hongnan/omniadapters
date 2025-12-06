from __future__ import annotations

import instructor

from ...core.constants import ANTHROPIC_IMPORT_ERROR

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message as AnthropicResponse
except ImportError as e:
    raise ImportError(ANTHROPIC_IMPORT_ERROR) from e

from ...core.models import AnthropicProviderConfig
from .base import BaseAdapter


class AnthropicAdapter(BaseAdapter[AnthropicProviderConfig, AsyncAnthropic, AnthropicResponse]):
    def _create_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(**self.provider_config.get_client_kwargs())

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: AsyncAnthropic = self.client
        return instructor.from_anthropic(client, mode=self.instructor_config.mode)
