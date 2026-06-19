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

"""Same as ``01_demo.py`` but with ``make_trace_recorder`` collecting events.

Parity proof for ``structify``'s ``with_hooks=True`` / ``CompletionTrace``.
The recorder is a tiny utility — pydantic-ai's own ``event_stream_handler``
parameter does the work.

```bash
uv run playground/pydantic_ai/text/04_traced.py --provider openai
```
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel

from omniadapters.pydantic_ai import make_trace_recorder
from playground.pydantic_ai.text.settings import DemoProvider, create_demo_adapter

console = Console()


class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


async def traced_review(provider: DemoProvider, movie: str) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            output_type=MovieReview,
            instructions="You are a helpful movie critic.",
        )

        handler, trace = make_trace_recorder()
        result = await agent.run(
            f"Review the movie '{movie}' for me.",
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
            event_stream_handler=handler,
        )

        console.print(Panel.fit(f"🎬 {result.output.title}  ⭐ {result.output.rating}/10", style="bold blue"))
        console.print(result.output.summary)

        kinds = Counter(type(e).__name__ for e in trace.events)
        console.print(
            Panel.fit(
                "\n".join(f"  • {k}: {v}" for k, v in kinds.most_common()) or "(no events)",
                title="📊 Captured event-stream summary",
                border_style="cyan",
            )
        )
        console.print(
            f"[dim]all_messages: {len(result.all_messages())} | "
            f"input_tokens={result.usage().input_tokens} "
            f"output_tokens={result.usage().output_tokens}[/dim]"
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai traced demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    parser.add_argument("--movie", default="Inception")
    args = parser.parse_args()
    await traced_review(args.provider, args.movie)


if __name__ == "__main__":
    asyncio.run(main())
