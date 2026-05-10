# /// script
# dependencies = [
#   "pydantic==2.12.0",
#   "pydantic-settings==2.11.0",
# ]
# ///

"""Settings for the pydantic-ai image demos. Same env grammar as the text demos
but a separate ``.env`` so callers can pin a vision-capable model."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from omniadapters.pydantic_ai import PydanticAIAdapter, create_adapter
from playground.pydantic_ai.text.settings import (
    CompletionParams,
    DemoProvider,
    TextModelFamily,
)


class Settings(BaseSettings):
    models: TextModelFamily

    model_config = SettingsConfigDict(
        env_file="playground/pydantic_ai/image/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(_env_file="playground/pydantic_ai/image/.env")  # pyright: ignore[reportCallIssue]


def create_demo_adapter(provider: DemoProvider) -> tuple[PydanticAIAdapter, CompletionParams]:
    family = getattr(get_settings().models, provider)
    adapter = create_adapter(provider_config=family.provider, model_name=family.completion.model)
    return adapter, family.completion
