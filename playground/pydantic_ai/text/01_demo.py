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

"""pydantic-ai parity demo for ``playground/structify/text/01_demo.py``.

```bash
uv run playground/pydantic_ai/text/01_demo.py --provider all
uv run playground/pydantic_ai/text/01_demo.py --provider openai --movie "Blade Runner"
uv run playground/pydantic_ai/text/01_demo.py --provider gemini --movie "Dune"
```

Demonstrates pydantic-ai's native ``Agent.run`` returning a structured Pydantic
model. The adapter we provide is the minimal ``PydanticAIAdapter``; everything
else is plain pydantic-ai surface.
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from playground.pydantic_ai.text.settings import (
    CompletionParams,
    DemoProvider,
    create_demo_adapter,
)

console = Console()


class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


def _settings_for(params: CompletionParams) -> ModelSettings:
    return ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens)


def create_review_table(review: MovieReview, provider_name: str) -> Table:
    table = Table(
        title=f"🎬 {review.title} - via {provider_name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Aspect", style="cyan", width=12)
    table.add_column("Details", style="white")
    table.add_row("Rating", f"⭐ {review.rating}/10")
    table.add_row("Summary", review.summary)
    table.add_row("Pros", "\n".join(f"✅ {pro}" for pro in review.pros))
    table.add_row("Cons", "\n".join(f"❌ {con}" for con in review.cons))
    return table


async def review_movie(provider: DemoProvider, movie: str) -> MovieReview:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            output_type=MovieReview,
            instructions="You are a helpful movie critic. Produce concise, structured reviews.",
        )
        console.print(Panel.fit(f"🤖 {provider.title()} Example", style="bold blue"))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"Getting review from {provider.title()}...", total=None)
            result = await agent.run(
                f"Review the movie '{movie}' for me.",
                model_settings=_settings_for(params),
            )
        console.print(create_review_table(result.output, provider.title()))
        console.print(
            f"[dim]usage: input={result.usage().input_tokens} "
            f"output={result.usage().output_tokens}[/dim]"
        )
        return result.output
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai movie review demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
    )
    parser.add_argument("--movie", default="Inception")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "🎬 [bold]pydantic-ai Demo[/bold] 🎬\n[dim]Native Agent.run with structured output[/dim]",
            style="bold green",
        )
    )

    providers: list[DemoProvider] = (
        ["openai", "anthropic", "gemini"]
        if args.provider == "all"
        else [args.provider]  # type: ignore[list-item]
    )

    for i, provider in enumerate(providers):
        await review_movie(provider, args.movie)
        if i < len(providers) - 1:
            console.print("\n" + "=" * 50 + "\n")

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("✅ [bold green]Demo Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
