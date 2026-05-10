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

"""Plain text completion via ``Agent.run`` (no ``output_type``).

Parity proof for ``omniadapters.completion.Adapter.agenerate(...)``: pydantic-ai
returns a ``str`` output when no schema is provided. Reuses the env file from
``playground/pydantic_ai/text/.env``.

```bash
uv run playground/pydantic_ai/chat/01_basic_chat.py --provider openai \\
    --prompt "Explain recursion in two sentences."
```
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from playground.pydantic_ai.text.settings import DemoProvider, create_demo_adapter

console = Console()


async def chat(provider: DemoProvider, prompt: str) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            instructions="Always start your response with 'Hello, world!'",
        )
        result = await agent.run(
            prompt,
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
        )
        body = Text(result.output)
        body.append(
            f"\n\nModel: {result.usage().request_tokens} req / "
            f"{result.usage().response_tokens} resp tokens",
            style="dim italic",
        )
        console.print(
            Panel(body, title=f"💬 {provider.title()} Response", border_style="green", padding=(1, 2))
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai basic-chat demo")
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
    await chat(args.provider, args.prompt)


if __name__ == "__main__":
    asyncio.run(main())
