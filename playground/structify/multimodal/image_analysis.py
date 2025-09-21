# /// script
# dependencies = [
#   "openai==1.105.0",
#   "rich",
# ]
# ///
"""
Image Analysis Demo - Multi-provider Vision Intelligence

This demo showcases image analysis capabilities across different LLM providers,
extracting structured information including objects, text, charts, and insights.

Usage:

    # Analyze URL image (example with instructor's test image)
    uv run playground/structify/multimodal/image_analysis.py \
        --image-url https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg \
        --provider openai
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, cast

from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.structify import create_adapter
from omniadapters.structify.adapters.anthropic import AnthropicAdapter
from omniadapters.structify.adapters.gemini import GeminiAdapter
from omniadapters.structify.adapters.openai import OpenAIAdapter
from omniadapters.structify.models import InstructorConfig
from playground.structify.multimodal.models import ImageAnalysis
from playground.structify.multimodal.utils import display_analysis_results

console = Console()


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"

    anthropic_api_key: str
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-exp"

    model_config = SettingsConfigDict(
        env_file="playground/structify/multimodal/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def create_demo_adapter(
    provider: Literal["openai", "anthropic", "gemini"],
    settings: Settings,
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
    match provider:
        case "openai":
            return create_adapter(
                provider_config=OpenAIProviderConfig(api_key=settings.openai_api_key),
                completion_params=OpenAICompletionClientParams(model=settings.openai_model),
                instructor_config=InstructorConfig(mode=Mode.TOOLS),
            )
        case "anthropic":
            return create_adapter(
                provider_config=AnthropicProviderConfig(api_key=settings.anthropic_api_key),
                completion_params=AnthropicCompletionClientParams(
                    model=settings.anthropic_model,
                ),
                instructor_config=InstructorConfig(mode=Mode.ANTHROPIC_TOOLS),
            )
        case "gemini":
            return create_adapter(
                provider_config=GeminiProviderConfig(api_key=settings.gemini_api_key),
                completion_params=GeminiCompletionClientParams(model=settings.gemini_model),
                instructor_config=InstructorConfig(mode=Mode.GEMINI_JSON),
            )


async def analyze_image(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    image_url: str,
    provider_name: str,
    mode: str = "general",
    show_progress: bool = True,
) -> ImageAnalysis:
    system_prompt = {
        "role": "system",
        "content": "You are an expert image analyst. Provide detailed, structured analysis of images.",
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ],
    }

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            system_prompt,
            user_message,
        ],
    )

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(
                description=f"🔍 Analyzing image with {provider_name}...",
                total=None,
            )

            if provider_name.lower() == "anthropic":
                analysis = await adapter.acreate(
                    messages=messages,
                    response_model=ImageAnalysis,
                    max_tokens=4096,
                )
            else:
                analysis = await adapter.acreate(
                    messages=messages,
                    response_model=ImageAnalysis,
                )
    else:
        if provider_name.lower() == "anthropic":
            analysis = await adapter.acreate(
                messages=messages,
                response_model=ImageAnalysis,
                max_tokens=4096,
            )
        else:
            analysis = await adapter.acreate(
                messages=messages,
                response_model=ImageAnalysis,
            )

    analysis.provider_metadata["provider"] = provider_name
    analysis.provider_metadata["mode"] = mode

    return analysis


async def batch_analyze(
    pattern: str,
    provider: Literal["openai", "anthropic", "gemini"],
    settings: Settings,
    mode: str = "general",
) -> dict[str, ImageAnalysis]:
    path = Path(pattern)

    if path.is_dir():
        files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
    else:
        files = list(Path(".").glob(pattern))

    if not files:
        console.print(f"[red]No images found matching: {pattern}[/red]")
        return {}

    console.print(f"[green]Found {len(files)} images to analyze[/green]")

    adapter = create_demo_adapter(provider, settings)
    results = {}

    try:
        for i, file_path in enumerate(files, 1):
            console.print(f"\n[cyan]Processing {i}/{len(files)}: {file_path.name}[/cyan]")
            analysis = await analyze_image(
                adapter,
                file_path.as_uri(),
                provider,
                mode,
                show_progress=True,
            )
            results[str(file_path)] = analysis
            display_analysis_results(analysis, provider)
    finally:
        await adapter.aclose()

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-provider image analysis")
    parser.add_argument(
        "--image-url",
        type=str,
        help="Path or URL to image file",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Batch process images matching pattern",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--mode",
        choices=["general", "ocr", "chart", "objects", "scene"],
        default="general",
        help="Analysis mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results across providers",
    )
    parser.add_argument(
        "--export",
        choices=["json", "markdown"],
        help="Export format for results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for exports",
    )

    args = parser.parse_args()

    if not args.image_url and not args.batch:
        parser.error("Either --image or --batch must be specified")

    try:
        settings = Settings()  # pyright: ignore
    except Exception as e:
        console.print(
            f"[red]Failed to load settings. Create a .env file in playground/structify/multimodal/[/red]\n{e}"
        )
        return

    console.print(
        Panel.fit(
            f"📸 [bold]Image Analysis Demo[/bold] 📸\n"
            f"[dim]Mode: {args.mode.title()} | Provider: {args.provider.title()}[/dim]",
            style="bold cyan",
        )
    )

    if args.batch:
        batch_provider = cast(
            Literal["openai", "anthropic", "gemini"],
            args.provider if args.provider != "all" else "openai",
        )
        results = await batch_analyze(
            args.batch,
            batch_provider,
            settings,
            args.mode,
        )
        if args.export:
            export_path = args.output or "batch_results.json"
            with open(export_path, "w") as f:
                json.dump(
                    {k: v.model_dump() for k, v in results.items()},
                    f,
                    indent=2,
                    default=str,
                )
            console.print(f"[green]Results exported to: {export_path}[/green]")
        return

    providers: list[Literal["openai", "anthropic", "gemini"]] = (
        ["openai", "anthropic", "gemini"] if args.provider == "all" else [args.provider]
    )

    results = {}

    for i, provider in enumerate(providers):
        if i > 0:
            console.print("\n" + "=" * 50 + "\n")

        try:
            adapter = create_demo_adapter(provider, settings)
            try:
                console.print(f"\n[bold blue]🤖 {provider.title()} Analysis[/bold blue]")
                analysis = await analyze_image(
                    adapter,
                    args.image_url,
                    provider.title(),
                    args.mode,
                )
                results[provider] = analysis

                display_analysis_results(
                    analysis,
                    provider.title(),
                    show_json=(args.export == "json"),
                )

            finally:
                await adapter.aclose()

        except Exception as e:
            console.print(f"[red]❌ Error with {provider}: {e}[/red]")
            continue

    console.print(Panel.fit("✅ [bold green]Analysis Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
