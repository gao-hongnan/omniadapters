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

"""Multimodal *non-structured* chat: image in, text out.

Closes the parity matrix between ``completion/`` (text-only chat) and the
multimodal ``structify/`` image demo by showing pydantic-ai handles both at
once with the same ``Agent`` surface.

```bash
uv run playground/pydantic_ai/chat/03_multimodal_chat.py \\
    --image-url https://raw.githubusercontent.com/gao-hongnan/omniadapters/006f8e3a27ca19a7401f44f32c882b63b3a56e37/playground/assets/chiwawa.png \\
    --provider openai
```
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic_ai import ImageUrl
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel

from playground.pydantic_ai.image.settings import create_demo_adapter
from playground.pydantic_ai.text.settings import DemoProvider


console = Console()


async def describe(image_url: str, provider: DemoProvider) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            instructions="Describe images in plain prose. No bullet points.",
        )
        result = await agent.run(
            ["Please describe this image.", ImageUrl(url=image_url)],
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
        )
        console.print(
            Panel(result.output, title=f"💬 {provider.title()} Description", border_style="green", padding=(1, 2))
        )
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai multimodal-chat demo")
    parser.add_argument("--image-url", required=True)
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
    )
    args = parser.parse_args()
    await describe(args.image_url, args.provider)


if __name__ == "__main__":
    asyncio.run(main())
