from __future__ import annotations

from typing import TYPE_CHECKING, assert_never, overload

from omniadapters.core.models import (
    AnthropicProviderConfig,
    AzureOpenAIProviderConfig,
    GeminiProviderConfig,
    OpenAIProviderConfig,
)

if TYPE_CHECKING:
    from omniadapters.completion.adapters.anthropic import AnthropicAdapter
    from omniadapters.completion.adapters.azure_openai import AzureOpenAIAdapter
    from omniadapters.completion.adapters.gemini import GeminiAdapter
    from omniadapters.completion.adapters.openai import OpenAIAdapter
    from omniadapters.core.models import (
        AnthropicCompletionClientParams,
        AzureOpenAICompletionClientParams,
        CompletionClientParams,
        GeminiCompletionClientParams,
        OpenAICompletionClientParams,
        ProviderConfig,
    )


@overload
def create_adapter(
    *,
    provider_config: OpenAIProviderConfig,
    completion_params: OpenAICompletionClientParams,
) -> OpenAIAdapter: ...


@overload
def create_adapter(
    *,
    provider_config: AnthropicProviderConfig,
    completion_params: AnthropicCompletionClientParams,
) -> AnthropicAdapter: ...


@overload
def create_adapter(
    *,
    provider_config: GeminiProviderConfig,
    completion_params: GeminiCompletionClientParams,
) -> GeminiAdapter: ...


@overload
def create_adapter(
    *,
    provider_config: AzureOpenAIProviderConfig,
    completion_params: AzureOpenAICompletionClientParams,
) -> AzureOpenAIAdapter: ...


@overload
def create_adapter(
    *,
    provider_config: ProviderConfig,
    completion_params: CompletionClientParams,
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter | AzureOpenAIAdapter: ...


def create_adapter(
    *,
    provider_config: ProviderConfig,
    completion_params: CompletionClientParams,
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter | AzureOpenAIAdapter:
    match provider_config:
        case OpenAIProviderConfig():
            from omniadapters.completion.adapters.openai import OpenAIAdapter

            return OpenAIAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case AnthropicProviderConfig():
            from omniadapters.completion.adapters.anthropic import AnthropicAdapter

            return AnthropicAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case GeminiProviderConfig():
            from omniadapters.completion.adapters.gemini import GeminiAdapter

            return GeminiAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case AzureOpenAIProviderConfig():
            from omniadapters.completion.adapters.azure_openai import AzureOpenAIAdapter

            return AzureOpenAIAdapter(
                provider_config=provider_config,
                completion_params=completion_params,
            )
        case _:
            assert_never(provider_config)
