# /// script
# dependencies = [
#   "openai==1.105.0",
#   "anthropic==0.66.0",
#   "google-genai==1.33.0",
#   "instructor==1.10.0",
#   "jsonref==1.1.0",
#   "pydantic==2.11.7",
#   "pydantic-settings==2.10.1",
#   "rich==14.1.0",
# ]
# ///

"""
Multi-provider image analysis with structured extraction.

Usage:
    # Default (all providers)
    uv run playground/structify/image/image_analysis.py --image-url <url>

    # Specific provider
    uv run playground/structify/image/image_analysis.py --image-url <url> --provider openai|anthropic|gemini

    # With streaming and/or tracing
    uv run playground/structify/image/image_analysis.py --image-url <url> --stream --trace

Example URL: https://raw.githubusercontent.com/gao-hongnan/omniadapters/006f8e3a27ca19a7401f44f32c882b63b3a56e37/playground/assets/chiwawa.png
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Literal, Union, assert_never, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.pretty import pprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from omniadapters.structify.adapters.anthropic import AnthropicAdapter
from omniadapters.structify.adapters.gemini import GeminiAdapter
from omniadapters.structify.adapters.openai import OpenAIAdapter
from omniadapters.structify.models import CompletionResult
from playground.structify.image.settings import create_demo_adapter

console = Console()

SYSTEM_MESSAGE = "You are an expert image analyst. Provide detailed, structured analysis of images."
USER_PROMPT = "Analyze this image in detail. Identify all objects, text, charts, scene elements, and provide insights."


class BoundingBox(BaseModel):
    x: float = Field(..., description="X coordinate (0-1 normalized)")
    y: float = Field(..., description="Y coordinate (0-1 normalized)")
    width: float = Field(..., description="Width (0-1 normalized)")
    height: float = Field(..., description="Height (0-1 normalized)")


class DetectedObject(BaseModel):
    label: str = Field(..., description="Object label/class")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bounding_box: Union[BoundingBox, None] = Field(None, description="Object location")


class ImageAnalysis(BaseModel):
    description: str = Field(..., description="Overall image description")
    detected_objects: list[DetectedObject] = Field(default_factory=list, description="Detected objects")
    confidence_level: str = Field(default="medium", description="Overall confidence level")


def build_provider_message_content(image_url: str, provider: Literal["openai", "anthropic"]) -> Any:
    """Build provider-specific message content for image analysis"""
    match provider:
        case "openai":
            return [{"type": "text", "text": USER_PROMPT}, {"type": "image_url", "image_url": {"url": image_url}}]
        case "anthropic":
            return [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image", "source": {"type": "url", "url": image_url}},
            ]
        case _:
            assert_never(provider)


def build_analysis_messages(
    image_url: str, provider: Literal["openai", "anthropic", "gemini"]
) -> list[ChatCompletionMessageParam]:
    """Build messages for image analysis across all providers"""
    if provider == "gemini":
        from instructor.multimodal import Image

        return cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        USER_PROMPT,
                        Image.from_url(image_url),
                    ],
                },
            ],
        )
    else:
        return cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_provider_message_content(image_url, provider)},
            ],
        )


def create_analysis_table(analysis: ImageAnalysis, provider_name: str) -> Table:
    table = Table(
        title=f"📸 {provider_name} Image Analysis",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Aspect", style="cyan", width=20)
    table.add_column("Details", style="white")

    table.add_row("Description", analysis.description)

    if analysis.detected_objects:
        objects_str = "\n".join(f"✓ {obj.label} ({obj.confidence:.2f})" for obj in analysis.detected_objects[:5])
        if len(analysis.detected_objects) > 5:
            objects_str += f"\n... and {len(analysis.detected_objects) - 5} more"
        table.add_row("Detected Objects", objects_str)

    table.add_row("Confidence", analysis.confidence_level.replace("_", " ").title())

    return table


def format_streaming_text(partial: ImageAnalysis) -> Text:
    text = Text()

    if hasattr(partial, "description") and partial.description:
        text.append("📷 Description:\n", style="bold cyan")
        text.append(f"{partial.description}\n", style="white")
        text.append("\n")

    if hasattr(partial, "detected_objects") and partial.detected_objects:
        text.append("🎯 Detected Objects:\n", style="bold green")
        for obj in partial.detected_objects[:5]:
            text.append(f"   • {obj.label}", style="green")
            if obj.confidence:
                text.append(f" ({obj.confidence:.2f})", style="dim green")
            text.append("\n")
        text.append("\n")

    return text


def display_trace_info(result: CompletionResult[Any, Any]) -> None:
    json_display = Syntax(
        code=result.trace.model_dump_json(indent=4, fallback=lambda x: str(x)),
        lexer="json",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )

    console.print(
        Panel(
            json_display,
            title="📊 [bold cyan]Trace Information[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=False,
        )
    )


async def analyze_image(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    image_url: str,
    provider: Literal["openai", "anthropic", "gemini"],
    show_trace: bool = False,
) -> ImageAnalysis:
    console.print(Panel.fit(f"🤖 {provider.title()} Analysis", style="bold blue"))

    messages = build_analysis_messages(image_url, provider)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Analyzing image with {provider.title()}...", total=None)

        if show_trace:
            result = await adapter.acreate(
                messages=messages,
                response_model=ImageAnalysis,
                with_hooks=True,
            )
            pprint(type(result.trace.raw_response))
            analysis = result.data
            display_trace_info(result)
        else:
            analysis = await adapter.acreate(
                messages=messages,
                response_model=ImageAnalysis,
            )
            pprint(analysis)

    console.print(create_analysis_table(analysis, provider.title()))
    return analysis


async def analyze_image_streaming(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    image_url: str,
    provider: Literal["openai", "anthropic", "gemini"],
) -> ImageAnalysis:
    console.print(Panel.fit(f"🤖 {provider.title()} Streaming Analysis", style="bold blue"))
    console.print("[dim]Streaming updates...[/dim]\n")

    messages = build_analysis_messages(image_url, provider)

    partial_count = 0
    final_analysis = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial_analysis in adapter.astream(
            messages=messages,
            response_model=ImageAnalysis,
        ):
            partial_count += 1
            final_analysis = partial_analysis

            formatted = format_streaming_text(partial_analysis)
            live.update(formatted)

            await asyncio.sleep(0.02)

    console.print(f"\n[green]✓ Streaming complete! Received {partial_count} partial updates[/green]")
    if final_analysis:
        console.print("\n[bold]Final Result:[/bold]")
        console.print(create_analysis_table(final_analysis, provider.title()))

    return final_analysis or ImageAnalysis(description="Unknown", confidence_level="low", detected_objects=[])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-provider image analysis with streaming support")
    parser.add_argument(
        "--image-url",
        type=str,
        required=True,
        help="URL of the image to analyze",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="LLM provider to use (default: all)",
    )
    parser.add_argument("--trace", action="store_true", help="Show trace information")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "📸 [bold]Image Analysis Demo[/bold] 📸\n[dim]Multi-provider vision intelligence with streaming[/dim]",
            style="bold green",
        )
    )

    providers: list[Literal["openai", "anthropic", "gemini"]] = (
        ["openai", "anthropic", "gemini"] if args.provider == "all" else [args.provider]
    )

    for i, provider in enumerate(providers):
        adapter = create_demo_adapter(provider)
        try:
            if args.stream:
                await analyze_image_streaming(adapter, args.image_url, provider)
            else:
                await analyze_image(adapter, args.image_url, provider, show_trace=args.trace)
        finally:
            await adapter.aclose()

        if i < len(providers) - 1:
            console.print("\n" + "=" * 50 + "\n")

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("✅ [bold green]Analysis Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
