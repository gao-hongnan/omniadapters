"""Real upstream completion error-handling smoke-test demo.

This intentionally sends bad requests through the normal omniadapters completion
adapters so you can verify vendor SDK errors are mapped into
`CompletionHTTPError` / `CompletionAPIError`.

Examples:
```bash
uv run playground/completion/04_error_handling.py --provider all --scenario invalid-model
uv run playground/completion/04_error_handling.py --provider all --scenario invalid-token-limit
uv run playground/completion/04_error_handling.py --provider openai --scenario invalid-message-role
uv run playground/completion/04_error_handling.py --provider openai --scenario invalid-api-key
```

"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING, Literal, assert_never, cast

from _shared import (
    CompletionDemoAdapter,
    CompletionPlaygroundSettings,
    PlaygroundConfigurationError,
    ProviderName,
    console,
    load_settings,
    parse_provider_selection,
    provider_display_name,
    render_completion_error,
    render_configuration_error,
    render_unexpected_error,
    selected_providers,
)
from pydantic import SecretStr
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError
from omniadapters.completion.factory import create_adapter
from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionClientParams,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
    ProviderConfig,
)

if TYPE_CHECKING:
    from omniadapters.core.types import MessageParam

type ErrorScenario = Literal["invalid-model", "invalid-api-key", "invalid-token-limit", "invalid-message-role"]

INVALID_MODELS: dict[ProviderName, str] = {
    "openai": "omniadapters-invalid-openai-model",
    "anthropic": "omniadapters-invalid-anthropic-model",
    "gemini": "omniadapters-invalid-gemini-model",
}

INVALID_API_KEYS: dict[ProviderName, str] = {
    "openai": "sk-omniadapters-real-upstream-error-demo-invalid-key",
    "anthropic": "sk-ant-api03-omniadapters-real-upstream-error-demo-invalid-key",
    "gemini": "AIzaSyDomniadaptersRealUpstreamErrorDemoInvalidKey000000",
}


def scenario_description(scenario: ErrorScenario) -> str:
    """Return a concise scenario description for terminal output."""
    match scenario:
        case "invalid-model":
            return "Calls the real upstream API with your configured key and a deliberately invalid model name."
        case "invalid-api-key":
            return "Calls the real upstream API with a deliberately invalid API key and the configured model name."
        case "invalid-token-limit":
            return "Calls the real upstream API with a provider-specific token limit set to an invalid negative value."
        case "invalid-message-role":
            return (
                "Calls the real upstream API with a malformed message role while preserving the configured model/key."
            )
        case _:
            assert_never(scenario)


def scenario_prompt(scenario: ErrorScenario) -> str:
    """Return the prompt used for the intentionally failing request."""
    match scenario:
        case "invalid-model":
            return "Reply with one sentence. This request should fail because the model name is invalid."
        case "invalid-api-key":
            return "Reply with one sentence. This request should fail because the API key is invalid."
        case "invalid-token-limit":
            return "Reply with one sentence. This request should fail because the token limit is invalid."
        case "invalid-message-role":
            return "Reply with one sentence. This request should fail because the message role is invalid."
        case _:
            assert_never(scenario)


def provider_config_from_settings(
    provider: ProviderName,
    settings: CompletionPlaygroundSettings,
) -> ProviderConfig:
    """Read provider credentials from completion playground settings."""
    match provider:
        case "openai":
            return settings.models.openai.provider
        case "anthropic":
            return settings.models.anthropic.provider
        case "gemini":
            return settings.models.gemini.provider
        case _:
            assert_never(provider)


def completion_params_from_settings(
    provider: ProviderName,
    settings: CompletionPlaygroundSettings,
) -> CompletionClientParams:
    """Read completion parameters from completion playground settings."""
    match provider:
        case "openai":
            return settings.models.openai.completion
        case "anthropic":
            return settings.models.anthropic.completion
        case "gemini":
            return settings.models.gemini.completion
        case _:
            assert_never(provider)


def provider_config_for_scenario(
    provider: ProviderName,
    scenario: ErrorScenario,
    provider_config: ProviderConfig,
) -> ProviderConfig:
    """Apply scenario-specific provider configuration overrides."""
    if scenario != "invalid-api-key":
        return provider_config

    match provider_config:
        case OpenAIProviderConfig():
            return provider_config.model_copy(update={"api_key": SecretStr(INVALID_API_KEYS[provider])})
        case AnthropicProviderConfig():
            return provider_config.model_copy(update={"api_key": SecretStr(INVALID_API_KEYS[provider])})
        case GeminiProviderConfig():
            return provider_config.model_copy(update={"api_key": SecretStr(INVALID_API_KEYS[provider])})
        case _:
            msg = f"Unsupported provider config for error demo: {type(provider_config).__name__}"
            raise PlaygroundConfigurationError(msg)


def require_configured_api_key(provider: ProviderName, provider_config: ProviderConfig) -> None:
    """Require a real configured key for scenarios that should preserve credentials."""
    if provider_config.api_key.get_secret_value().strip():
        return

    env_name = f"MODELS__{provider.upper()}__PROVIDER__API_KEY"
    msg = f"{env_name} must be set to test upstream errors with configured credentials."
    raise PlaygroundConfigurationError(msg)


def completion_params_for_scenario(
    provider: ProviderName,
    scenario: ErrorScenario,
    completion_params: CompletionClientParams,
) -> CompletionClientParams:
    """Apply scenario-specific completion parameter overrides."""
    if scenario != "invalid-model":
        return completion_params

    match completion_params:
        case OpenAICompletionClientParams():
            return completion_params.model_copy(update={"model": INVALID_MODELS[provider]})
        case AnthropicCompletionClientParams():
            return completion_params.model_copy(update={"model": INVALID_MODELS[provider]})
        case GeminiCompletionClientParams():
            return completion_params.model_copy(update={"model": INVALID_MODELS[provider]})
        case _:
            msg = f"Unsupported completion params for error demo: {type(completion_params).__name__}"
            raise PlaygroundConfigurationError(msg)


def request_kwargs_for_scenario(provider: ProviderName, scenario: ErrorScenario) -> dict[str, object]:
    """Return per-call request overrides that intentionally trigger upstream validation errors."""
    if scenario != "invalid-token-limit":
        return {}

    match provider:
        case "openai":
            return {"max_completion_tokens": -1}
        case "anthropic":
            return {"max_tokens": -1}
        case "gemini":
            return {"max_output_tokens": -1}
        case _:
            assert_never(provider)


def create_scenario_adapter(
    provider: ProviderName,
    scenario: ErrorScenario,
    settings: CompletionPlaygroundSettings,
) -> CompletionDemoAdapter:
    """Create an adapter configured to trigger one real upstream failure."""
    provider_config = provider_config_for_scenario(
        provider,
        scenario,
        provider_config_from_settings(provider, settings),
    )
    if scenario != "invalid-api-key":
        require_configured_api_key(provider, provider_config)

    completion_params = completion_params_for_scenario(
        provider,
        scenario,
        completion_params_from_settings(provider, settings),
    )
    return cast(
        "CompletionDemoAdapter",
        create_adapter(
            provider_config=provider_config,
            completion_params=completion_params,
        ),
    )


def scenario_table(provider: ProviderName, scenario: ErrorScenario, settings: CompletionPlaygroundSettings) -> Table:
    """Render the exact upstream failure scenario about to run."""
    completion_params = completion_params_for_scenario(
        provider,
        scenario,
        completion_params_from_settings(provider, settings),
    )
    request_kwargs = request_kwargs_for_scenario(provider, scenario)

    table = Table(title=f"{provider_display_name(provider)} upstream error scenario", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Scenario", scenario)
    table.add_row(
        "Expected mapping",
        "CompletionHTTPError for upstream 4xx/5xx, CompletionAPIError for transport/API failures",
    )
    table.add_row("Model sent", completion_params.model)
    table.add_row("API key", "deliberately invalid" if scenario == "invalid-api-key" else "configured key")
    table.add_row("Request overrides", json.dumps(request_kwargs) if request_kwargs else "{}")
    return table


async def run_provider_scenario(
    provider: ProviderName,
    scenario: ErrorScenario,
    settings: CompletionPlaygroundSettings,
) -> None:
    """Run one provider/scenario pair and render the captured completion error."""
    adapter = create_scenario_adapter(provider, scenario, settings)
    if scenario == "invalid-message-role":
        messages: list[MessageParam] = [
            {"role": "definitely_not_a_valid_chat_role", "content": scenario_prompt(scenario)}
        ]
    else:
        messages = [{"role": "user", "content": scenario_prompt(scenario)}]
    request_kwargs = request_kwargs_for_scenario(provider, scenario)

    console.print(scenario_table(provider, scenario, settings))
    try:
        response = await adapter.agenerate(messages=messages, **request_kwargs)
    except CompletionHTTPError as error:
        render_completion_error(error)
    except CompletionAPIError as error:
        render_completion_error(error)
    else:
        message = (
            f"Request unexpectedly succeeded with model {response.model}. "
            "The upstream provider did not reject this scenario."
        )
        console.print(
            Panel(
                Text(message, style="yellow"),
                title="Unexpected success",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    finally:
        await adapter.aclose()


def parse_scenario(value: str) -> ErrorScenario:
    """Validate a scenario from argparse."""
    match value:
        case "invalid-model" | "invalid-api-key" | "invalid-token-limit" | "invalid-message-role":
            return value
        case _:
            msg = f"Unsupported error scenario: {value}"
            raise PlaygroundConfigurationError(msg)


async def main() -> None:
    """Parse CLI options and run real upstream error smoke tests."""
    parser = argparse.ArgumentParser(description="Real upstream completion error-handling smoke-test demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="Provider to exercise",
    )
    parser.add_argument(
        "--scenario",
        choices=["invalid-model", "invalid-api-key", "invalid-token-limit", "invalid-message-role"],
        default="invalid-token-limit",
        help="Real upstream failure scenario to trigger",
    )
    args = parser.parse_args()

    provider_selection = parse_provider_selection(args.provider)
    scenario = parse_scenario(args.scenario)
    settings = load_settings()

    console.print(
        Panel.fit(
            f"Completion upstream error-handling demo\n[dim]{scenario_description(scenario)}[/dim]",
            border_style="red",
        )
    )

    providers = selected_providers(provider_selection)
    for index, provider in enumerate(providers):
        await run_provider_scenario(provider, scenario, settings)
        if index < len(providers) - 1:
            console.rule(style="dim")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Error-handling demo interrupted.[/yellow]")
    except PlaygroundConfigurationError as error:
        render_configuration_error(error)
    except Exception as error:  # noqa: BLE001 - CLI boundary renders unexpected smoke-test failures.
        render_unexpected_error(error)
