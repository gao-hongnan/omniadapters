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

"""Multi-turn conversation via ``message_history``.

Parity proof for ``structify``'s manual ``conversation_history.append(...)``
pattern: pydantic-ai threads ``result.all_messages()`` back into the next call.

```bash
uv run playground/pydantic_ai/text/02_conversation.py --provider openai
```
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from playground.pydantic_ai.text.settings import DemoProvider, create_demo_adapter

console = Console()


class IssueAnalysis(BaseModel):
    problem_category: Literal["hardware", "software", "network", "performance", "other"]
    severity: Literal["low", "medium", "high", "critical"]
    symptoms: list[str] = Field(description="List of symptoms reported by user")
    possible_causes: list[str] = Field(description="Potential root causes")
    confidence: float = Field(ge=0, le=1)


class ConversationResponse(BaseModel):
    message: str
    needs_more_info: bool = False
    follow_up_questions: list[str] = Field(default_factory=list)


async def tech_support(provider: DemoProvider) -> None:
    adapter, params = create_demo_adapter(provider)
    settings = ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens)
    try:
        chat_agent = adapter.create_agent(
            output_type=ConversationResponse,
            instructions=(
                "You are a helpful tech support assistant. Help users diagnose and "
                "solve their computer issues. Be patient and ask clarifying questions."
            ),
        )
        analysis_agent = adapter.create_agent(
            output_type=IssueAnalysis,
            instructions="Analyze the technical issue based on the conversation so far.",
        )

        console.print(
            Panel.fit(
                f"🔧 [bold cyan]Tech Support Assistant[/bold cyan] — {provider.title()}",
                border_style="cyan",
            )
        )

        message_history = None
        user_messages = [
            "My computer has been running really slow lately",
            "It started about a week ago. I notice it especially when opening programs",
            "Yes, the fans are running loudly and it feels warm",
            "I haven't cleaned it in about 6 months",
        ]

        analysis: IssueAnalysis | None = None
        for i, user_msg in enumerate(user_messages, 1):
            console.print(f"\n[cyan]User:[/cyan] {user_msg}")
            result = await chat_agent.run(
                user_msg,
                message_history=message_history,
                model_settings=settings,
            )
            console.print(f"[green]Assistant:[/green] {result.output.message}")
            if result.output.follow_up_questions:
                console.print("\n[dim]Follow-up questions:[/dim]")
                for q in result.output.follow_up_questions:
                    console.print(f"  [dim]• {q}[/dim]")
            message_history = result.all_messages()

            if i == 2:
                # Side query that re-uses the same history but with a different output_type.
                console.print("\n[yellow]📊 Analyzing issue...[/yellow]")
                ar = await analysis_agent.run(
                    "Summarize the issue so far.",
                    message_history=message_history,
                    model_settings=settings,
                )
                analysis = ar.output

                table = Table(title="Issue Analysis", show_header=False)
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("Category", analysis.problem_category)
                table.add_row("Severity", analysis.severity)
                table.add_row("Confidence", f"{analysis.confidence:.0%}")
                table.add_row("Symptoms", "\n".join(f"• {s}" for s in analysis.symptoms))
                table.add_row("Possible Causes", "\n".join(f"• {c}" for c in analysis.possible_causes))
                console.print(table)

        console.print("\n" + "=" * 50)
        console.print(
            Panel.fit(
                f"[green]Conversation Complete![/green]\n"
                f"Messages exchanged: {len(user_messages)}\n"
                f"Issue identified: {analysis.problem_category if analysis else 'N/A'}",
                style="green",
            )
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai conversation demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    args = parser.parse_args()
    await tech_support(args.provider)


if __name__ == "__main__":
    asyncio.run(main())
