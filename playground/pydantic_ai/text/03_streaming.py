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

"""Streaming structured output via ``Agent.run_stream``.

Parity proof for ``structify.astream`` — partial Pydantic instances arrive as
the model emits text, rendered live with rich.

```bash
uv run playground/pydantic_ai/text/03_streaming.py --provider openai
uv run playground/pydantic_ai/text/03_streaming.py --provider anthropic --movie "Dune"
```
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from playground.pydantic_ai.text.settings import DemoProvider, create_demo_adapter

console = Console()


class MovieReview(BaseModel):
    title: str = ""
    rating: float = Field(default=0.0, ge=0, le=10)
    summary: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


def format_partial(partial: MovieReview) -> Text:
    text = Text()
    if partial.title:
        text.append(f"🎬 {partial.title}\n", style="bold cyan")
    if partial.rating:
        text.append(f"⭐ {partial.rating}/10\n", style="yellow")
    if partial.summary:
        text.append("\nSummary: ", style="bold magenta")
        text.append(f"{partial.summary}\n", style="white")
    if partial.pros:
        text.append("\n✅ Pros:\n", style="bold green")
        for pro in partial.pros:
            text.append(f"   • {pro}\n", style="green")
    if partial.cons:
        text.append("\n❌ Cons:\n", style="bold red")
        for con in partial.cons:
            text.append(f"   • {con}\n", style="red")
    return text


async def stream_review(provider: DemoProvider, movie: str) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            output_type=MovieReview,
            instructions="You are a helpful movie critic.",
        )

        console.print(Panel.fit(f"🤖 {provider.title()} Streaming Example", style="bold blue"))
        console.print("[dim]Streaming partial Pydantic models...[/dim]\n")

        partial_count = 0
        async with agent.run_stream(
            f"Review the movie '{movie}' for me.",
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
        ) as run:
            with Live(console=console, refresh_per_second=30, transient=False) as live:
                async for partial in run.stream_output():
                    partial_count += 1
                    live.update(format_partial(partial))
                    await asyncio.sleep(0.02)

        console.print(
            f"\n[green]✓ Streaming complete — received {partial_count} partial updates[/green]"
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai streaming demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    parser.add_argument("--movie", default="Inception")
    args = parser.parse_args()
    await stream_review(args.provider, args.movie)


if __name__ == "__main__":
    asyncio.run(main())
