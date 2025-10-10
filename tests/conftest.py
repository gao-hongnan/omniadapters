from __future__ import annotations

import instructor
import pytest
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.structify.adapters.anthropic import AnthropicAdapter
from omniadapters.structify.adapters.gemini import GeminiAdapter
from omniadapters.structify.adapters.openai import OpenAIAdapter
from omniadapters.structify.models import InstructorConfig


class TestSettings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI__API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC__API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI__API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SimpleTestModel(BaseModel):
    name: str = Field(description="A name")
    age: int = Field(description="An age", ge=0, le=150)
    description: str = Field(description="A brief description")


class ComplexTestModel(BaseModel):
    title: str = Field(description="Title of the content")
    rating: float = Field(description="Rating from 0 to 10", ge=0, le=10)
    tags: list[str] = Field(description="List of relevant tags")
    summary: str = Field(description="Brief summary of the content")
    is_recommended: bool = Field(description="Whether this is recommended")


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    return TestSettings()


@pytest.fixture(scope="session")
def skip_if_no_api_keys(test_settings: TestSettings) -> None:
    if not all(
        [
            test_settings.openai_api_key,
            test_settings.anthropic_api_key,
            test_settings.gemini_api_key,
        ]
    ):
        pytest.skip("API keys not available for integration tests")


def _has_all_api_keys() -> bool:
    """Check if all required API keys are available."""
    settings = TestSettings()
    return all(
        [
            settings.openai_api_key,
            settings.anthropic_api_key,
            settings.gemini_api_key,
        ]
    )


skip_without_api_keys = pytest.mark.skipif(
    not _has_all_api_keys(), reason="API keys not available for integration tests"
)


@pytest.fixture
def openai_provider_config(test_settings: TestSettings) -> OpenAIProviderConfig:
    return OpenAIProviderConfig(api_key=SecretStr(test_settings.openai_api_key))


@pytest.fixture
def anthropic_provider_config(test_settings: TestSettings) -> AnthropicProviderConfig:
    return AnthropicProviderConfig(api_key=SecretStr(test_settings.anthropic_api_key))


@pytest.fixture
def gemini_provider_config(test_settings: TestSettings) -> GeminiProviderConfig:
    return GeminiProviderConfig(api_key=SecretStr(test_settings.gemini_api_key))


@pytest.fixture
def openai_completion_params() -> OpenAICompletionClientParams:
    return OpenAICompletionClientParams.model_validate(
        {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_completion_tokens": 500,
        }
    )


@pytest.fixture
def anthropic_completion_params() -> AnthropicCompletionClientParams:
    return AnthropicCompletionClientParams.model_validate(
        {
            "model": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "max_tokens": 500,
        }
    )


@pytest.fixture
def gemini_completion_params() -> GeminiCompletionClientParams:
    return GeminiCompletionClientParams.model_validate(
        {
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "max_output_tokens": 500,
        }
    )


@pytest.fixture
def openai_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.TOOLS)


@pytest.fixture
def anthropic_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)


@pytest.fixture
def gemini_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)


@pytest.fixture
def openai_adapter(
    openai_provider_config: OpenAIProviderConfig,
    openai_completion_params: OpenAICompletionClientParams,
    openai_instructor_config: InstructorConfig,
) -> OpenAIAdapter:
    adapter = OpenAIAdapter(
        provider_config=openai_provider_config,
        completion_params=openai_completion_params,
        instructor_config=openai_instructor_config,
    )
    return adapter


@pytest.fixture
def anthropic_adapter(
    anthropic_provider_config: AnthropicProviderConfig,
    anthropic_completion_params: AnthropicCompletionClientParams,
    anthropic_instructor_config: InstructorConfig,
) -> AnthropicAdapter:
    adapter = AnthropicAdapter(
        provider_config=anthropic_provider_config,
        completion_params=anthropic_completion_params,
        instructor_config=anthropic_instructor_config,
    )
    return adapter


@pytest.fixture
def gemini_adapter(
    gemini_provider_config: GeminiProviderConfig,
    gemini_completion_params: GeminiCompletionClientParams,
    gemini_instructor_config: InstructorConfig,
) -> GeminiAdapter:
    adapter = GeminiAdapter(
        provider_config=gemini_provider_config,
        completion_params=gemini_completion_params,
        instructor_config=gemini_instructor_config,
    )
    return adapter


@pytest.fixture
def simple_test_model_class() -> type[SimpleTestModel]:
    return SimpleTestModel


@pytest.fixture
def complex_test_model_class() -> type[ComplexTestModel]:
    return ComplexTestModel


@pytest.fixture
def test_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides structured responses.",
        },
        {
            "role": "user",
            "content": "Generate information about a person named Alice who is 25 years old and works as a software engineer.",
        },
    ]


@pytest.fixture
def complex_test_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful movie reviewer that provides detailed structured reviews.",
        },
        {
            "role": "user",
            "content": "Review the movie 'Inception' with a rating, tags, and metadata.",
        },
    ]


pytest_plugins = ["pytest_asyncio"]


async def cleanup_adapter(adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter) -> None:
    await adapter.aclose()


@pytest.fixture(autouse=True)
async def auto_cleanup_adapters(request: pytest.FixtureRequest):
    """Automatically cleanup adapters after each test."""
    yield

    for fixture_name in ["openai_adapter", "anthropic_adapter", "gemini_adapter"]:
        if fixture_name in request.fixturenames:
            try:
                adapter = request.getfixturevalue(fixture_name)
                if adapter:
                    await cleanup_adapter(adapter)
            except (RuntimeError, Exception):
                pass


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring API keys",
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test that uses mocking",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _ = config
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
