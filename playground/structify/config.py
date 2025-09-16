"""
Shared configuration for structify demos
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generic, Literal, TypeVar, assert_never

import instructor
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


class OpenAICompletion(OpenAICompletionClientParams):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)
    max_completion_tokens: int = Field(default=1000)


class AnthropicCompletion(AnthropicCompletionClientParams):
    model: str = Field(default="claude-3-5-haiku-20241022")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class GeminiCompletion(GeminiCompletionClientParams):
    model: str = Field(default="gemini-2.5-flash", exclude=True)
    temperature: float = Field(default=1.0)
    max_output_tokens: int = Field(default=1000)


class InstructorSettings(BaseModel):
    mode: str = Field(default="TOOLS")

    def get_mode(self) -> instructor.Mode:
        """Convert string mode to instructor.Mode enum"""
        mode_map = {
            "TOOLS": instructor.Mode.TOOLS,
            "TOOLS_STRICT": instructor.Mode.TOOLS_STRICT,
            "ANTHROPIC_TOOLS": instructor.Mode.ANTHROPIC_TOOLS,
            "GENAI_STRUCTURED_OUTPUTS": instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        }
        return mode_map.get(self.mode, instructor.Mode.TOOLS)


P = TypeVar("P", OpenAIProviderConfig, AnthropicProviderConfig, GeminiProviderConfig)
C = TypeVar("C", OpenAICompletion, AnthropicCompletion, GeminiCompletion)


class ProviderFamily(BaseModel, Generic[P, C]):
    provider: P
    completion: C
    instructor: InstructorSettings


class ModelFamily(BaseModel):
    """Container for all provider configurations"""

    openai: ProviderFamily[OpenAIProviderConfig, OpenAICompletion]
    anthropic: ProviderFamily[AnthropicProviderConfig, AnthropicCompletion]
    gemini: ProviderFamily[GeminiProviderConfig, GeminiCompletion]

    def create_adapter(
        self, provider: Literal["openai", "anthropic", "gemini"]
    ) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
        """Create adapter for specified provider"""
        match provider:
            case "openai":
                return create_adapter(
                    provider_config=self.openai.provider,
                    completion_params=self.openai.completion,
                    instructor_config=InstructorConfig(mode=self.openai.instructor.get_mode()),
                )
            case "anthropic":
                return create_adapter(
                    provider_config=self.anthropic.provider,
                    completion_params=self.anthropic.completion,
                    instructor_config=InstructorConfig(mode=self.anthropic.instructor.get_mode()),
                )
            case "gemini":
                return create_adapter(
                    provider_config=self.gemini.provider,
                    completion_params=self.gemini.completion,
                    instructor_config=InstructorConfig(mode=self.gemini.instructor.get_mode()),
                )
            case _:
                assert_never(provider)


class Settings(BaseSettings):
    models: ModelFamily

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=128)
def get_settings() -> Settings:
    """Get the singleton settings instance"""
    return Settings(_env_file="playground/structify/.env")  # pyright: ignore[reportCallIssue]


def create_demo_adapter(
    provider: Literal["openai", "anthropic", "gemini"],
) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
    """Create an adapter for the specified provider"""
    settings = get_settings()
    return settings.models.create_adapter(provider)
