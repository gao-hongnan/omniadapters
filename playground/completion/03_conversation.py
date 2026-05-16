"""Multi-turn conversation smoke-test demo.

Examples:
```bash
uv run playground/completion/03_conversation.py --provider anthropic
uv run playground/completion/03_conversation.py --provider openai --stream
uv run playground/completion/03_conversation.py --provider gemini --interactive
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
    parse_provider_name,
    provider_display_name,
    render_completion_error,
    render_configuration_error,
    render_unexpected_error,
    streaming_text,
)
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError

if TYPE_CHECKING:
    from omniadapters.core.types import MessageParam

SYSTEM_PROMPT = (
    "You are a concise bookstore assistant. Keep track of the customer's stated preferences, "
    "recommend specific books, and explain each recommendation in one or two sentences."
)

DEMO_CUSTOMER_MESSAGES: tuple[str, ...] = (
    "Hi. I want something new to read. I enjoyed the Harry Potter series.",
    "I liked the magical school setting and the coming-of-age arc. What should I try next?",
    "I also like mysteries. Can you recommend a magical mystery?",
    "Give me one pick for a thirteen-year-old and one pick for an adult.",
)


async def stream_response(
    adapter: CompletionDemoAdapter,
    messages: list[MessageParam],
    *,
    speaker: str,
) -> str:
    """Stream one assistant response and return the complete content."""
    console.print(f"\n[bold blue]{speaker}:[/bold blue]")

    chunk_count = 0
    full_content = ""
    stream = await adapter.agenerate(messages=messages, stream=True)

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        async for chunk in stream:
            chunk_count += 1
            full_content += chunk.content
            live.update(streaming_text(full_content, chunk_count))

    return full_content


async def normal_response(
    adapter: CompletionDemoAdapter,
    messages: list[MessageParam],
    *,
    speaker: str,
) -> str:
    """Generate one non-streaming assistant response."""
    response = await adapter.agenerate(messages=messages)
    content = response.content or ""
    console.print(f"\n[bold blue]{speaker}:[/bold blue] [cyan]{content}[/cyan]")
    return content


async def assistant_response(
    adapter: CompletionDemoAdapter,
    messages: list[MessageParam],
    *,
    speaker: str,
    stream: bool,
) -> str:
    """Generate one assistant response in the selected mode."""
    if stream:
        return await stream_response(adapter, messages, speaker=speaker)
    return await normal_response(adapter, messages, speaker=speaker)


async def run_interactive_conversation(
    adapter: CompletionDemoAdapter,
    conversation_history: list[MessageParam],
    *,
    stream: bool,
) -> None:
    """Run an interactive bookstore conversation."""
    console.print("[yellow]Interactive mode. Type 'quit', 'exit', or 'q' to stop.[/yellow]")

    while True:
        user_input = Prompt.ask("\n[bold green]You[/bold green]")
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        conversation_history.append({"role": "user", "content": user_input})
        response = await assistant_response(adapter, conversation_history, speaker="Bookstore Assistant", stream=stream)
        conversation_history.append({"role": "assistant", "content": response})


async def run_scripted_conversation(
    adapter: CompletionDemoAdapter,
    conversation_history: list[MessageParam],
    *,
    stream: bool,
) -> None:
    """Run the scripted bookstore conversation."""
    for customer_message in DEMO_CUSTOMER_MESSAGES:
        console.print(f"\n[bold green]Customer:[/bold green] {customer_message}")
        conversation_history.append({"role": "user", "content": customer_message})
        response = await assistant_response(adapter, conversation_history, speaker="Bookstore Assistant", stream=stream)
        conversation_history.append({"role": "assistant", "content": response})

        if not stream:
            await asyncio.sleep(0.25)


async def bookstore_conversation_demo(
    provider: ProviderName,
    *,
    stream: bool,
    interactive: bool,
) -> None:
    """Run a multi-turn bookstore assistant smoke test."""
    adapter = create_provider_adapter(provider)
    conversation_history: list[MessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        console.print(
            Panel.fit(
                f"Bookstore assistant conversation\n[dim]Provider: {provider_display_name(provider)}[/dim]",
                border_style="cyan",
            )
        )

        if interactive:
            await run_interactive_conversation(adapter, conversation_history, stream=stream)
        else:
            await run_scripted_conversation(adapter, conversation_history, stream=stream)

        user_message_count = sum(1 for message in conversation_history if message["role"] == "user")
        console.print(
            Panel.fit(
                f"[green]Conversation complete[/green]\n"
                f"Provider: {provider_display_name(provider)}\n"
                f"User messages: {user_message_count}\n"
                f"Mode: {'interactive' if interactive else 'scripted'} / {'streaming' if stream else 'standard'}",
                border_style="green",
            )
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    """Parse CLI options and run the conversation demo."""
    parser = argparse.ArgumentParser(description="Multi-turn bookstore conversation smoke-test demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="anthropic",
        help="LLM provider to use",
    )
    parser.add_argument("--stream", action="store_true", help="Stream assistant responses")
    parser.add_argument("--interactive", action="store_true", help="Use an interactive conversation")
    args = parser.parse_args()

    try:
        await bookstore_conversation_demo(
            parse_provider_name(args.provider),
            stream=args.stream,
            interactive=args.interactive,
        )
    except CompletionHTTPError as error:
        render_completion_error(error)
    except CompletionAPIError as error:
        render_completion_error(error)
    except PlaygroundConfigurationError as error:
        render_configuration_error(error)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversation demo interrupted.[/yellow]")
    except Exception as error:  # noqa: BLE001 - CLI boundary renders unexpected smoke-test failures.
        render_unexpected_error(error)
