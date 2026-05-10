# /// script
# dependencies = [
#   "pydantic==2.12.0",
#   "pydantic-settings==2.11.0",
# ]
# ///

"""Shared settings for the pydantic-ai text demos.

The env layout mirrors ``playground/structify/text/.env.sample`` so the same
``.env`` keys work — minus the ``__INSTRUCTOR__MODE`` line, which pydantic-ai
chooses automatically based on the provider and ``output_type`` strategy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omniadapters.core.models import (
    AnthropicProviderConfig,
    GeminiProviderConfig,
    OpenAIProviderConfig,
)
from omniadapters.pydantic_ai import PydanticAIAdapter, create_adapter

DemoProvider = Literal["openai", "anthropic", "gemini"]


class CompletionParams(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)


class ProviderFamily(BaseModel):
    provider: OpenAIProviderConfig | AnthropicProviderConfig | GeminiProviderConfig
    completion: CompletionParams


class TextModelFamily(BaseModel):
    openai: ProviderFamily
    anthropic: ProviderFamily
    gemini: ProviderFamily

    def create(self, provider: DemoProvider) -> tuple[PydanticAIAdapter, CompletionParams]:
        family: ProviderFamily = getattr(self, provider)
        adapter = create_adapter(
            provider_config=family.provider,
            model_name=family.completion.model,
        )
        return adapter, family.completion


class Settings(BaseSettings):
    models: TextModelFamily

    model_config = SettingsConfigDict(
        env_file="playground/pydantic_ai/text/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_env_file="playground/pydantic_ai/text/.env")  # pyright: ignore[reportCallIssue]


def create_demo_adapter(provider: DemoProvider) -> tuple[PydanticAIAdapter, CompletionParams]:
    return get_settings().models.create(provider)
