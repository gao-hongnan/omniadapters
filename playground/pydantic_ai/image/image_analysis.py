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

"""Multimodal structured extraction via pydantic-ai's ``ImageUrl``.

Parity proof for ``playground/structify/image/image_analysis.py``: same
``ImageAnalysis`` schema, but using pydantic-ai's normalized image input
instead of provider-specific shapes.

```bash
uv run playground/pydantic_ai/image/image_analysis.py \\
    --image-url https://raw.githubusercontent.com/gao-hongnan/omniadapters/006f8e3a27ca19a7401f44f32c882b63b3a56e37/playground/assets/chiwawa.png \\
    --provider openai
```
"""

from __future__ import annotations

import argparse
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import ImageUrl
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from playground.pydantic_ai.image.settings import create_demo_adapter
from playground.pydantic_ai.text.settings import DemoProvider

console = Console()

SYSTEM_MESSAGE = "You are an expert image analyst. Provide detailed, structured analysis of images."
USER_PROMPT = "Analyze this image in detail. Identify all objects and provide insights."


class BoundingBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox | None = None


class ImageAnalysis(BaseModel):
    description: str
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    confidence_level: str = "medium"


def render(analysis: ImageAnalysis, provider_name: str) -> Table:
    table = Table(
        title=f"📸 {provider_name} Image Analysis",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Aspect", style="cyan", width=20)
    table.add_column("Details", style="white")
    table.add_row("Description", analysis.description)
    if analysis.detected_objects:
        objs = "\n".join(f"✓ {o.label} ({o.confidence:.2f})" for o in analysis.detected_objects[:5])
        if len(analysis.detected_objects) > 5:
            objs += f"\n... and {len(analysis.detected_objects) - 5} more"
        table.add_row("Detected Objects", objs)
    table.add_row("Confidence", analysis.confidence_level)
    return table


async def analyze(image_url: str, provider: DemoProvider) -> None:
    adapter, params = create_demo_adapter(provider)
    try:
        agent = adapter.create_agent(
            output_type=ImageAnalysis,
            instructions=SYSTEM_MESSAGE,
        )
        console.print(Panel.fit(f"🤖 {provider.title()} Analysis", style="bold blue"))

        result = await agent.run(
            [USER_PROMPT, ImageUrl(url=image_url)],
            model_settings=ModelSettings(temperature=params.temperature, max_tokens=params.max_tokens),
        )
        console.print(render(result.output, provider.title()))
    finally:
        await adapter.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="pydantic-ai multimodal demo")
    parser.add_argument("--image-url", required=True)
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
    )
    args = parser.parse_args()

    providers: list[DemoProvider] = (
        ["openai", "anthropic", "gemini"]
        if args.provider == "all"
        else [args.provider]  # type: ignore[list-item]
    )
    for i, p in enumerate(providers):
        await analyze(args.image_url, p)
        if i < len(providers) - 1:
            console.print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
