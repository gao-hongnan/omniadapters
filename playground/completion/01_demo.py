"""Completion Adapter Pattern Demo - Multi-Provider LLM Completions

```bash
uv run playground/completion/01_demo.py --provider all

uv run playground/completion/01_demo.py --provider openai
uv run playground/completion/01_demo.py --provider anthropic
uv run playground/completion/01_demo.py --provider gemini
uv run playground/completion/01_demo.py --provider azure-openai

uv run playground/completion/01_demo.py --prompt "Explain quantum computing"
uv run playground/completion/01_demo.py --provider anthropic --prompt "Tell me a joke"

uv run playground/completion/01_demo.py --stream
uv run playground/completion/01_demo.py --provider openai --stream
uv run playground/completion/01_demo.py --provider gemini --prompt "What is AI?" --stream

uv run playground/completion/01_demo.py --trace --provider all
uv run playground/completion/01_demo.py --provider openai --trace
uv run playground/completion/01_demo.py --provider openai --prompt "Hello world" --stream --trace
```
"""

from __future__ import annotations

import argparse
import asyncio
from pprint import pprint
from typing import Generic, Literal, TypeVar, assert_never

from anthropic.types import Message
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from omniadapters.completion.adapters.anthropic import AnthropicAdapter
from omniadapters.completion.adapters.azure_openai import AzureOpenAIAdapter
from omniadapters.completion.adapters.gemini import GeminiAdapter
from omniadapters.completion.adapters.openai import OpenAIAdapter
from omniadapters.completion.factory import create_adapter
from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionResponse,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.core.types import MessageParam

console = Console()


class OpenAICompletion(OpenAICompletionClientParams):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)
    max_completion_tokens: int = Field(default=1000)


class AnthropicCompletion(AnthropicCompletionClientParams):
    model: str = Field(default="claude-3-5-haiku-20241022")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class GeminiCompletion(GeminiCompletionClientParams):
    model: str = Field(default="gemini-2.0-flash-exp", exclude=True)
    temperature: float = Field(default=1.0)
    max_output_tokens: int = Field(default=1000)


P = TypeVar("P", OpenAIProviderConfig, AnthropicProviderConfig, GeminiProviderConfig)
C = TypeVar("C", OpenAICompletion, AnthropicCompletion, GeminiCompletion)


class ProviderFamily(BaseModel, Generic[P, C]):
    provider: P
    completion: C


class ModelFamily(BaseModel):
    """Container for all provider configurations"""

    openai: ProviderFamily[OpenAIProviderConfig, OpenAICompletion]
    anthropic: ProviderFamily[AnthropicProviderConfig, AnthropicCompletion]
    gemini: ProviderFamily[GeminiProviderConfig, GeminiCompletion]

    def create_adapter(
        self, provider: Literal["openai", "anthropic", "gemini"]
    ) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
        """Create adapter for specified provider"""
        match provider:
            case "openai":
                return create_adapter(provider_config=self.openai.provider, completion_params=self.openai.completion)
            case "anthropic":
                return create_adapter(
                    provider_config=self.anthropic.provider, completion_params=self.anthropic.completion
                )
            case "gemini":
                return create_adapter(provider_config=self.gemini.provider, completion_params=self.gemini.completion)
            case _:
                assert_never(provider)


class Settings(BaseSettings):
    models: ModelFamily

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


settings = Settings(_env_file="playground/completion/.env")  # pyright: ignore[reportCallIssue]


def create_response_panel(
    response: CompletionResponse[ChatCompletion]
    | CompletionResponse[Message]
    | CompletionResponse[GenerateContentResponse],
    provider_name: str,
) -> Panel:
    """Create a formatted panel for completion response."""
    content = Text(response.content or "")

    metadata = []
    if response.model:
        metadata.append(f"Model: {response.model}")
    if response.usage:
        metadata.append(
            f"Tokens: {response.usage.prompt_tokens} prompt + "
            f"{response.usage.completion_tokens} completion = "
            f"{response.usage.total_tokens} total"
        )

    if metadata:
        content.append("\n\n", style="dim")
        content.append(" | ".join(metadata), style="dim italic")

    return Panel(
        content,
        title=f"💬 {provider_name} Response",
        border_style="green",
        padding=(1, 2),
    )


def format_streaming_text(content: str, chunk_count: int) -> Text:
    """Format streaming text with chunk counter."""
    text = Text()
    text.append(content, style="green")
    text.append(f"\n\n[dim]Chunks received: {chunk_count}[/dim]", style="dim")
    return text


def display_trace_info(
    response: CompletionResponse[ChatCompletion]
    | CompletionResponse[Message]
    | CompletionResponse[GenerateContentResponse],
) -> None:
    """Display trace information about the raw response."""
    raw_type = type(response.raw_response).__name__
    json_str = response.raw_response.model_dump_json(indent=2)

    json_display = Syntax(
        code=json_str,
        lexer="json",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )

    console.print(
        Panel(
            json_display,
            title=f"📊 [bold cyan]Trace Information - {raw_type}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=False,
        )
    )


async def generate_completion(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter | AzureOpenAIAdapter,
    messages: list[MessageParam],
    provider_name: str,
    show_trace: bool = False,
) -> None:
    """Generate a normal (non-streaming) completion."""
    console.print(Panel.fit(f"🤖 {provider_name} Example", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Getting response from {provider_name}...", total=None)

        response = await adapter.agenerate(messages=messages)

    console.print(create_response_panel(response, provider_name))

    if show_trace:
        pprint(type(response.raw_response))
        display_trace_info(response)


async def generate_completion_streaming(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter | AzureOpenAIAdapter,
    messages: list[MessageParam],
    provider_name: str,
) -> None:
    """Generate a streaming completion with real-time updates."""
    console.print(Panel.fit(f"🤖 {provider_name} Streaming Example", style="bold blue"))
    console.print("[dim]Streaming tokens...[/dim]\n")

    chunk_count = 0
    full_content = ""

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        stream = await adapter.agenerate(messages=messages, stream=True)
        async for chunk in stream:
            chunk_count += 1
            if chunk.content:
                full_content += chunk.content

            formatted = format_streaming_text(full_content, chunk_count)
            live.update(formatted)

            await asyncio.sleep(0.01)

    console.print(f"\n[green]✓ Streaming complete! Received {chunk_count} chunks[/green]")


def create_demo_adapter(
    provider: Literal["openai", "anthropic", "gemini"],
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter | AzureOpenAIAdapter:
    """Create an adapter for the specified provider."""
    return settings.models.create_adapter(provider)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-provider LLM completion demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="LLM provider to use (default: all)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain recursion in programming in 2 sentences.",
        help="Prompt to send (default: Explain recursion)",
    )
    parser.add_argument("--trace", action="store_true", help="Show trace information")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "🚀 [bold]Completion Demo[/bold] 🚀\n[dim]Unified interface for multiple LLM providers[/dim]",
            style="bold green",
        )
    )

    messages: list[MessageParam] = [
        {"role": "system", "content": "Always start your response with 'Hello, world!'"},
        {"role": "user", "content": args.prompt},
    ]

    console.print(f"\n📝 [yellow]Prompt:[/yellow] {args.prompt}\n")

    providers: list[Literal["openai", "anthropic", "gemini"]] = (
        ["openai", "anthropic", "gemini"] if args.provider == "all" else [args.provider.replace("-", " ").lower()]
    )

    for i, provider in enumerate(providers):
        try:
            adapter = create_demo_adapter(provider)
            try:
                if args.stream:
                    await generate_completion_streaming(adapter, messages, provider.replace("-", " ").title())
                else:
                    await generate_completion(
                        adapter, messages, provider.replace("-", " ").title(), show_trace=args.trace
                    )
            finally:
                await adapter.aclose()
        except Exception as e:
            console.print(f"[red]❌ Error with {provider}: {e}[/red]")

        if i < len(providers) - 1:
            console.print("\n" + "=" * 50 + "\n")

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("✅ [bold green]Demo Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
