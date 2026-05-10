# /// script
# dependencies = [
#   "openai==2.8.0",
#   "anthropic==0.72.0",
#   "google-genai==1.42.0",
#   "pydantic==2.12.0",
#   "pydantic-ai==1.93.0",
#   "pydantic-settings==2.11.0",
#   "rich==14.2.0",
# ]
# ///

"""Token-by-token text streaming via ``Agent.run_stream(...).stream_text(delta=True)``.

This is the parity proof against ``omniadapters.completion.Adapter.agenerate(stream=True)``.

```bash
uv run playground/pydantic_ai/chat/02_streaming_chat.py --provider openai
uv run playground/pydantic_ai/chat/02_streaming_chat.py --provider gemini --prompt "What is AI?"
```
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from playground.pydantic_ai.text.settings import DemoProvider, create_demo_adapter

console = Console()


async def stream_chat(provider: DemoProvider, prompt: str) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            instructions="Always start your response with 'Hello, world!'",
        )
        console.print(Panel.fit(f"🤖 {provider.title()} Streaming Example", style="bold blue"))
        console.print("[dim]Streaming tokens...[/dim]\n")

        chunk_count = 0
        full_text = ""
        async with agent.run_stream(
            prompt,
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
        ) as run:
            with Live(console=console, refresh_per_second=30, transient=False) as live:
                async for delta in run.stream_text(delta=True):
                    chunk_count += 1
                    full_text += delta
                    live.update(Text(full_text, style="green"))
                    await asyncio.sleep(0.01)

        console.print(f"\n[green]✓ Streaming complete — received {chunk_count} chunks[/green]")
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai streaming-chat demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    parser.add_argument(
        "--prompt",
        default="Explain recursion in programming in 2 sentences.",
    )
    args = parser.parse_args()
    await stream_chat(args.provider, args.prompt)


if __name__ == "__main__":
    asyncio.run(main())
