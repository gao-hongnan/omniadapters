# /// script
# dependencies = [
#   "instructor==1.10.0",
#   "pydantic==2.11.7",
#   "pydantic-settings==2.10.1",
# ]
# ///

from __future__ import annotations

from functools import lru_cache
from typing import Literal, cast

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
from playground.structify.settings import ProviderFamily


class OpenAICompletion(OpenAICompletionClientParams):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_completion_tokens: int = Field(default=4096, gt=0, le=100000)


class AnthropicCompletion(AnthropicCompletionClientParams):
    model: str = Field(default="claude-3-5-sonnet-20241022")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0, le=200000)


class GeminiCompletion(GeminiCompletionClientParams):
    model: str = Field(default="gemini-2.0-flash-exp", exclude=True)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, gt=0, le=8192)


class ImageModelFamily(BaseModel):
    openai: ProviderFamily[OpenAIProviderConfig, OpenAICompletion]
    anthropic: ProviderFamily[AnthropicProviderConfig, AnthropicCompletion]
    gemini: ProviderFamily[GeminiProviderConfig, GeminiCompletion]

    def create_adapter(
        self, provider: Literal["openai", "anthropic", "gemini"]
    ) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
        family: (
            ProviderFamily[OpenAIProviderConfig, OpenAICompletion]
            | ProviderFamily[AnthropicProviderConfig, AnthropicCompletion]
            | ProviderFamily[GeminiProviderConfig, GeminiCompletion]
        ) = getattr(self, provider)
        return cast(
            OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
            create_adapter(
                provider_config=family.provider,
                completion_params=family.completion,
                instructor_config=InstructorConfig(mode=family.instructor.mode),
            ),
        )


class Settings(BaseSettings):
    models: ImageModelFamily

    model_config = SettingsConfigDict(
        env_file="playground/structify/image/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_env_file="playground/structify/image/.env")  # pyright: ignore[reportCallIssue]


def create_demo_adapter(
    provider: Literal["openai", "anthropic", "gemini"],
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
    settings = get_settings()
    return settings.models.create_adapter(provider)
