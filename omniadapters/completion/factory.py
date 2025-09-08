from __future__ import annotations

from typing import Any, assert_never, overload

from omniadapters.completion.adapters.anthropic import AnthropicAdapter
from omniadapters.completion.adapters.azure_openai import AzureOpenAIAdapter
from omniadapters.completion.adapters.base import BaseAdapter
from omniadapters.completion.adapters.gemini import GeminiAdapter
from omniadapters.completion.adapters.openai import OpenAIAdapter
from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    AzureOpenAICompletionClientParams,
    AzureOpenAIProviderConfig,
    CompletionClientParams,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
    ProviderConfig,
)


@overload
def create_adapter(
    provider_config: OpenAIProviderConfig,
    completion_params: OpenAICompletionClientParams | None = None,
) -> OpenAIAdapter: ...


@overload
def create_adapter(
    provider_config: AnthropicProviderConfig,
    completion_params: AnthropicCompletionClientParams | None = None,
) -> AnthropicAdapter: ...


@overload
def create_adapter(
    provider_config: GeminiProviderConfig,
    completion_params: GeminiCompletionClientParams | None = None,
) -> GeminiAdapter: ...


@overload
def create_adapter(
    provider_config: AzureOpenAIProviderConfig,
    completion_params: AzureOpenAICompletionClientParams | None = None,
) -> AzureOpenAIAdapter: ...


def create_adapter(
    provider_config: ProviderConfig,
    completion_params: CompletionClientParams | None = None,
) -> BaseAdapter[Any, Any, Any, Any]:
    match provider_config:
        case OpenAIProviderConfig():
            return OpenAIAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case AnthropicProviderConfig():
            return AnthropicAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case GeminiProviderConfig():
            return GeminiAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case AzureOpenAIProviderConfig():
            return AzureOpenAIAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case _:
            assert_never(provider_config)
