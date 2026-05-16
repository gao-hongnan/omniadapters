"""Shared helpers for completion playground scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError
from omniadapters.completion.factory import create_adapter
from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionResponse,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
    Usage,
)

if TYPE_CHECKING:
    from anthropic.types import Message
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion

    from omniadapters.completion.adapters.anthropic import AnthropicAdapter
    from omniadapters.completion.adapters.gemini import GeminiAdapter
    from omniadapters.completion.adapters.openai import OpenAIAdapter

type ProviderName = Literal["openai", "anthropic", "gemini"]
type ProviderSelection = ProviderName | Literal["all"]
type CompletionDemoAdapter = "OpenAIAdapter | AnthropicAdapter | GeminiAdapter"
type CompletionDemoResponse = (
    CompletionResponse["ChatCompletion"] | CompletionResponse["Message"] | CompletionResponse["GenerateContentResponse"]
)

COMPLETION_ENV_FILE: Final = Path(__file__).with_name(".env")
ALL_PROVIDERS: Final[tuple[ProviderName, ...]] = ("openai", "anthropic", "gemini")

console = Console()


class PlaygroundConfigurationError(RuntimeError):
    """Raised when a playground script is missing required local configuration."""


class DemoOpenAIProviderConfig(OpenAIProviderConfig):
    """OpenAI provider settings with an empty default for provider-specific demos."""

    api_key: SecretStr = Field(default=SecretStr(""))


class DemoAnthropicProviderConfig(AnthropicProviderConfig):
    """Anthropic provider settings with an empty default for provider-specific demos."""

    api_key: SecretStr = Field(default=SecretStr(""))


class DemoGeminiProviderConfig(GeminiProviderConfig):
    """Gemini provider settings with an empty default for provider-specific demos."""

    api_key: SecretStr = Field(default=SecretStr(""))


class OpenAICompletion(OpenAICompletionClientParams):
    """Default OpenAI completion parameters for playground smoke tests."""

    model: str = Field(default="gpt-5.4-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_completion_tokens: int = Field(default=1000, gt=0)


class AnthropicCompletion(AnthropicCompletionClientParams):
    """Default Anthropic completion parameters for playground smoke tests."""

    model: str = Field(default="claude-3-5-haiku-latest")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, gt=0)


class GeminiCompletion(GeminiCompletionClientParams):
    """Default Gemini completion parameters for playground smoke tests."""

    model: str = Field(default="gemini-3-flash-preview")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1000, gt=0)


class OpenAIModelSettings(BaseModel):
    """OpenAI provider and completion settings loaded from `.env`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: DemoOpenAIProviderConfig = Field(default_factory=DemoOpenAIProviderConfig)
    completion: OpenAICompletion = Field(default_factory=OpenAICompletion)


class AnthropicModelSettings(BaseModel):
    """Anthropic provider and completion settings loaded from `.env`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: DemoAnthropicProviderConfig = Field(default_factory=DemoAnthropicProviderConfig)
    completion: AnthropicCompletion = Field(default_factory=AnthropicCompletion)


class GeminiModelSettings(BaseModel):
    """Gemini provider and completion settings loaded from `.env`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: DemoGeminiProviderConfig = Field(default_factory=DemoGeminiProviderConfig)
    completion: GeminiCompletion = Field(default_factory=GeminiCompletion)


class ModelSettings(BaseModel):
    """All provider settings used by completion playground scripts."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    openai: OpenAIModelSettings = Field(default_factory=OpenAIModelSettings)
    anthropic: AnthropicModelSettings = Field(default_factory=AnthropicModelSettings)
    gemini: GeminiModelSettings = Field(default_factory=GeminiModelSettings)


class CompletionPlaygroundSettings(BaseSettings):
    """Settings loaded from `playground/completion/.env`."""

    model_config = SettingsConfigDict(
        env_file=COMPLETION_ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    models: ModelSettings = Field(default_factory=ModelSettings)


def load_settings() -> CompletionPlaygroundSettings:
    """Load completion playground settings from `.env`."""
    return CompletionPlaygroundSettings()


def selected_providers(selection: ProviderSelection) -> list[ProviderName]:
    """Expand a provider CLI selection into concrete providers."""
    if selection == "all":
        return list(ALL_PROVIDERS)
    return [selection]


def provider_display_name(provider: ProviderName) -> str:
    """Return a human-readable provider label."""
    return provider.replace("-", " ").title()


def _require_api_key(provider: ProviderName, api_key: SecretStr) -> None:
    if api_key.get_secret_value().strip():
        return

    env_name = f"MODELS__{provider.upper()}__PROVIDER__API_KEY"
    message = f"{env_name} is not set in {COMPLETION_ENV_FILE}"
    raise PlaygroundConfigurationError(message)


def create_provider_adapter(
    provider: ProviderName,
    settings: CompletionPlaygroundSettings | None = None,
) -> CompletionDemoAdapter:
    """Create a completion adapter for a configured provider."""
    resolved_settings = settings or load_settings()

    match provider:
        case "openai":
            openai_settings = resolved_settings.models.openai
            _require_api_key(provider, openai_settings.provider.api_key)
            return create_adapter(
                provider_config=openai_settings.provider,
                completion_params=openai_settings.completion,
            )
        case "anthropic":
            anthropic_settings = resolved_settings.models.anthropic
            _require_api_key(provider, anthropic_settings.provider.api_key)
            return create_adapter(
                provider_config=anthropic_settings.provider,
                completion_params=anthropic_settings.completion,
            )
        case "gemini":
            gemini_settings = resolved_settings.models.gemini
            _require_api_key(provider, gemini_settings.provider.api_key)
            return create_adapter(
                provider_config=gemini_settings.provider,
                completion_params=gemini_settings.completion,
            )
        case _:
            assert_never(provider)


def create_openai_adapter(settings: CompletionPlaygroundSettings | None = None) -> OpenAIAdapter:
    """Create an OpenAI adapter with an OpenAI-specific static return type."""
    resolved_settings = settings or load_settings()
    openai_settings = resolved_settings.models.openai
    _require_api_key("openai", openai_settings.provider.api_key)
    return create_adapter(
        provider_config=openai_settings.provider,
        completion_params=openai_settings.completion,
    )


def usage_summary(usage: Usage | None) -> str | None:
    """Format token usage for concise terminal display."""
    if usage is None:
        return None

    return f"{usage.input_tokens} input + {usage.output_tokens} output = {usage.total_tokens} total"


def response_panel(response: CompletionDemoResponse, provider: ProviderName) -> Panel:
    """Create a formatted Rich panel for a completion response."""
    content = Text(response.content or "", style="white")
    metadata: list[str] = []

    if response.model:
        metadata.append(f"Model: {response.model}")
    if summary := usage_summary(response.usage):
        metadata.append(f"Tokens: {summary}")

    if metadata:
        content.append("\n\n")
        content.append(" | ".join(metadata), style="dim italic")

    return Panel(
        content,
        title=f"{provider_display_name(provider)} response",
        border_style="green",
        padding=(1, 2),
    )


def streaming_text(content: str, chunk_count: int) -> Text:
    """Format streaming text with a chunk counter."""
    text = Text(content, style="cyan")
    text.append(f"\n\nChunks received: {chunk_count}", style="dim")
    return text


def trace_panel(response: CompletionDemoResponse) -> Panel:
    """Create a trace panel for raw vendor response and normalized usage."""
    raw_type = type(response.raw_response).__name__
    raw_json = response.raw_response.model_dump_json(indent=2)
    renderables: list[Syntax | Text] = [
        Syntax(raw_json, "json", theme="monokai", word_wrap=True),
    ]

    if response.usage:
        usage_json = response.usage.model_dump_json(indent=2)
        renderables.extend(
            [
                Text("\nNormalized usage", style="bold cyan"),
                Syntax(usage_json, "json", theme="monokai", word_wrap=True),
            ]
        )

    return Panel(
        Group(*renderables),
        title=f"Trace information: {raw_type}",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )


def _json_syntax(value: object) -> Syntax:
    rendered = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    return Syntax(rendered, "json", theme="monokai", word_wrap=True)


def render_completion_error(error: CompletionHTTPError | CompletionAPIError) -> None:
    """Render a structured completion adapter error."""
    if isinstance(error, CompletionHTTPError):
        body = Group(
            Text(f"HTTP status: {error.status_code}", style="bold red"),
            Text(f"Model: {error.model_name}", style="red"),
            Text(str(error), style="red"),
            Text("\nVendor body", style="bold red"),
            _json_syntax(error.body),
        )
        title = "Completion HTTP error"
    else:
        body = Group(
            Text(f"Model: {error.model_name}", style="bold red"),
            Text(str(error), style="red"),
        )
        title = "Completion API error"

    console.print(Panel(body, title=title, border_style="red", padding=(1, 2)))


def render_configuration_error(error: PlaygroundConfigurationError) -> None:
    """Render local playground configuration failures."""
    console.print(
        Panel(
            Text(str(error), style="yellow"),
            title="Playground configuration error",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def render_unexpected_error(error: Exception) -> None:
    """Render an unexpected exception at the CLI boundary."""
    console.print(
        Panel(
            Text(f"{type(error).__name__}: {error}", style="red"),
            title="Unexpected playground error",
            border_style="red",
            padding=(1, 2),
        )
    )


def parse_provider_selection(value: str) -> ProviderSelection:
    """Validate a provider selection from argparse."""
    match value:
        case "openai" | "anthropic" | "gemini" | "all":
            return value
        case _:
            msg = f"Unsupported provider selection: {value}"
            raise PlaygroundConfigurationError(msg)


def parse_provider_name(value: str) -> ProviderName:
    """Validate a concrete provider name from argparse."""
    match value:
        case "openai" | "anthropic" | "gemini":
            return value
        case _:
            msg = f"Unsupported provider: {value}"
            raise PlaygroundConfigurationError(msg)


type JsonSchema = dict[str, object]


def model_json_schema(model: type[BaseModel]) -> JsonSchema:
    """Return a JSON schema object suitable for OpenAI tool definitions."""
    return model.model_json_schema()
