from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.constants import OPENAI_IMPORT_ERROR

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(OPENAI_IMPORT_ERROR) from e

from ...core.cost import GENAI_PRICES_PROFILE, UsageExtractionSpec
from ...core.enums import Provider
from ...core.models import OpenAIProviderConfig
from .._map_api_errors import _map_openai_errors
from ._openai_compatible import OpenAICompatibleAdapter

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class OpenAIAdapter(OpenAICompatibleAdapter[OpenAIProviderConfig, AsyncOpenAI]):
    _usage_spec: ClassVar[UsageExtractionSpec] = GENAI_PRICES_PROFILE[Provider.OPENAI]

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self.provider_config.get_client_kwargs())

    def _map_errors(self, *, model_name: str) -> AbstractContextManager[None]:
        return _map_openai_errors(model_name=model_name)
