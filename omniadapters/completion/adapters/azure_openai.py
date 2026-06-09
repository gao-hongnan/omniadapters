from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.constants import AZURE_OPENAI_IMPORT_ERROR

try:
    from openai import AsyncAzureOpenAI
except ImportError as e:
    raise ImportError(AZURE_OPENAI_IMPORT_ERROR) from e

from ...core.cost import GENAI_PRICES_PROFILE, UsageExtractionSpec
from ...core.enums import Provider
from ...core.models import AzureOpenAIProviderConfig
from .._map_api_errors import _map_azure_openai_errors
from ._openai_compatible import OpenAICompatibleAdapter

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class AzureOpenAIAdapter(OpenAICompatibleAdapter[AzureOpenAIProviderConfig, AsyncAzureOpenAI]):
    # NOTE: Azure responses carry the OpenAI ChatCompletion usage shape, and the
    # 'azure' spec's OpenAI URL + 'chat' flavor resolve to OpenAI's extractor,
    # which is identical to Azure's -- so the inherited response/stream mapping
    # applies unchanged. Only the client and error mapper differ.
    _usage_spec: ClassVar[UsageExtractionSpec] = GENAI_PRICES_PROFILE[Provider.AZURE_OPENAI]

    def _create_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(**self.provider_config.get_client_kwargs())

    def _map_errors(self, *, model_name: str) -> AbstractContextManager[None]:
        return _map_azure_openai_errors(model_name=model_name)
