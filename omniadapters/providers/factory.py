from __future__ import annotations

from typing import Any, assert_never, overload

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
from omniadapters.providers.adapters.anthropic import AnthropicAdapter
from omniadapters.providers.adapters.azure_openai import AzureOpenAIAdapter
from omniadapters.providers.adapters.base import BaseAdapter
from omniadapters.providers.adapters.gemini import GeminiAdapter
from omniadapters.providers.adapters.openai import OpenAIAdapter


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
    """Create an LLM adapter based on provider configuration.

    Args:
        provider_config: Provider-specific configuration
        completion_params: Default completion parameters

    Returns:
        Provider-specific adapter instance

    Raises:
        ValueError: If provider type is unknown

    Examples:
        >>> # Create OpenAI adapter
        >>> config = OpenAIProviderConfig(api_key="sk-...")
        >>> adapter = create_adapter(config)

        >>> # Create Anthropic adapter with default params
        >>> config = AnthropicProviderConfig(api_key="sk-ant-...")
        >>> params = AnthropicCompletionClientParams(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=2000
        ... )
        >>> adapter = create_adapter(config, params)

        >>> # Create Gemini adapter
        >>> config = GeminiProviderConfig(api_key="AIza...")
        >>> adapter = create_adapter(config)
    """
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
