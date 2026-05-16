"""Multi-provider completion smoke-test demo.

Examples:
```bash
uv run playground/completion/01_demo.py --provider all
uv run playground/completion/01_demo.py --provider openai --trace
uv run playground/completion/01_demo.py --provider anthropic --stream
uv run playground/completion/01_demo.py --provider gemini --prompt "Say hello in one sentence"
```

"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

from _shared import (
    CompletionDemoAdapter,
    PlaygroundConfigurationError,
    ProviderName,
    console,
    create_provider_adapter,
    parse_provider_selection,
    provider_display_name,
    render_completion_error,
    render_configuration_error,
    render_unexpected_error,
    response_panel,
    selected_providers,
    streaming_text,
    trace_panel,
)
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError

if TYPE_CHECKING:
    from omniadapters.core.types import MessageParam


async def generate_completion(
    adapter: CompletionDemoAdapter,
    messages: list[MessageParam],
    provider: ProviderName,
    *,
    show_trace: bool,
) -> None:
    """Generate and display one non-streaming completion."""
    display_name = provider_display_name(provider)
    console.print(Panel.fit(f"{display_name} completion", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description=f"Waiting for {display_name}...", total=None)
        response = await adapter.agenerate(messages=messages)

    console.print(response_panel(response, provider))

    if show_trace:
        console.print(trace_panel(response))


async def generate_streaming_completion(
    adapter: CompletionDemoAdapter,
    messages: list[MessageParam],
    provider: ProviderName,
) -> None:
    """Generate and display one streaming completion."""
    display_name = provider_display_name(provider)
    console.print(Panel.fit(f"{display_name} streaming completion", style="bold blue"))

    chunk_count = 0
    full_content = ""
    stream = await adapter.agenerate(messages=messages, stream=True)

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        async for chunk in stream:
            chunk_count += 1
            full_content += chunk.content
            live.update(streaming_text(full_content, chunk_count))

    console.print(f"[green]Streaming complete. Received {chunk_count} chunks.[/green]")


async def run_provider(
    provider: ProviderName,
    messages: list[MessageParam],
    *,
    stream: bool,
    trace: bool,
) -> None:
    """Run the demo for one provider and always close its adapter."""
    adapter = create_provider_adapter(provider)
    try:
        if stream:
            await generate_streaming_completion(adapter, messages, provider)
        else:
            await generate_completion(adapter, messages, provider, show_trace=trace)
    finally:
        await adapter.aclose()


async def main() -> None:
    """Parse CLI options and run completion smoke tests."""
    parser = argparse.ArgumentParser(description="Multi-provider LLM completion smoke-test demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--prompt",
        default="Explain recursion in programming in 20 concise sentences.",
        help="Prompt to send",
    )
    parser.add_argument("--trace", action="store_true", help="Show raw vendor response and normalized usage")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    selection = parse_provider_selection(args.provider)
    providers = selected_providers(selection)
    messages: list[MessageParam] = [
        {"role": "system", "content": "Answer clearly and keep the response compact."},
        {"role": "user", "content": args.prompt},
    ]

    console.print(
        Panel.fit(
            "Completion playground smoke test\n[dim]Unified adapters with structured error rendering[/dim]",
            border_style="green",
        )
    )
    console.print(f"[yellow]Prompt:[/yellow] {args.prompt}\n")

    for index, provider in enumerate(providers):
        try:
            await run_provider(provider, messages, stream=args.stream, trace=args.trace)
        except CompletionHTTPError as error:
            render_completion_error(error)
        except CompletionAPIError as error:
            render_completion_error(error)
        except PlaygroundConfigurationError as error:
            render_configuration_error(error)

        if index < len(providers) - 1:
            console.rule(style="dim")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Completion demo interrupted.[/yellow]")
    except Exception as error:  # noqa: BLE001 - CLI boundary renders unexpected smoke-test failures.
        render_unexpected_error(error)
