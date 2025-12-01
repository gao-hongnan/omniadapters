from __future__ import annotations

import pytest

from omniadapters.core.enums import Model, Provider
from omniadapters.core.tokens import (
    AnthropicTokenCounterAdapter,
    CharacterEstimatorAdapter,
    GeminiTokenCounterAdapter,
    OpenAITokenCounterAdapter,
    TokenCounterAdapter,
    create_token_counter,
)


class TestOpenAITokenCounterAdapter:
    def test_init_with_model_string(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        assert counter.model == "gpt-4o"

    def test_provider(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        assert counter.provider == Provider.OPENAI

    def test_count_tokens(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        count = counter.count_tokens("Hello world")
        assert count > 0

    def test_count_tokens_consistency(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        text = "This is a test sentence for token counting."
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        assert count1 == count2

    def test_count_tokens_unknown_model_fallback(self) -> None:
        counter = OpenAITokenCounterAdapter("unknown-model")
        count = counter.count_tokens("Hello world")
        assert count > 0

    @pytest.mark.asyncio
    async def test_acount_tokens(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        text = "Hello world"
        sync_count = counter.count_tokens(text)
        async_count = await counter.acount_tokens(text)
        assert async_count == sync_count


class TestAnthropicTokenCounterAdapter:
    def test_init(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229")
        assert counter.model == "claude-3-opus-20240229"

    def test_init_with_api_key(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229", api_key="test-key")
        assert counter.model == "claude-3-opus-20240229"

    def test_provider(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229")
        assert counter.provider == Provider.ANTHROPIC

    def test_sync_fallback_count_tokens(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229")
        count = counter.count_tokens("Hello world")
        assert count > 0

    @pytest.mark.asyncio
    async def test_acount_tokens_without_api_key(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229")
        text = "Hello world"
        sync_count = counter.count_tokens(text)
        async_count = await counter.acount_tokens(text)
        assert async_count == sync_count


class TestGeminiTokenCounterAdapter:
    def test_init(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash")
        assert counter.model == "gemini-2.0-flash"

    def test_init_with_api_key(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash", api_key="test-key")
        assert counter.model == "gemini-2.0-flash"

    def test_provider(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash")
        assert counter.provider == Provider.GEMINI

    def test_sync_fallback_count_tokens(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash")
        count = counter.count_tokens("Hello world")
        assert count > 0

    @pytest.mark.asyncio
    async def test_acount_tokens_without_api_key(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash")
        text = "Hello world"
        sync_count = counter.count_tokens(text)
        async_count = await counter.acount_tokens(text)
        assert async_count == sync_count


class TestCharacterEstimatorAdapter:
    DEFAULT_CHARS_PER_TOKEN = 4
    CUSTOM_CHARS_PER_TOKEN = 3

    def test_default_chars_per_token(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model")
        assert estimator.chars_per_token == self.DEFAULT_CHARS_PER_TOKEN

    def test_custom_chars_per_token(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model", chars_per_token=self.CUSTOM_CHARS_PER_TOKEN)
        assert estimator.chars_per_token == self.CUSTOM_CHARS_PER_TOKEN

    def test_count_tokens(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model", chars_per_token=self.DEFAULT_CHARS_PER_TOKEN)
        count = estimator.count_tokens("12345678")
        expected_tokens = 8 // self.DEFAULT_CHARS_PER_TOKEN
        assert count == expected_tokens

    def test_provider_is_none(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model")
        assert estimator.provider is None

    @pytest.mark.asyncio
    async def test_acount_tokens(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model")
        text = "12345678"
        sync_count = estimator.count_tokens(text)
        async_count = await estimator.acount_tokens(text)
        assert async_count == sync_count


class TestCreateTokenCounter:
    def test_returns_openai_adapter_for_gpt_model(self) -> None:
        counter = create_token_counter(Model.GPT_4O)
        assert isinstance(counter, OpenAITokenCounterAdapter)

    def test_returns_openai_adapter_for_gpt_string(self) -> None:
        counter = create_token_counter("gpt-4o")
        assert isinstance(counter, OpenAITokenCounterAdapter)

    def test_returns_anthropic_adapter_for_claude_model(self) -> None:
        counter = create_token_counter(Model.CLAUDE_SONNET_4_5)
        assert isinstance(counter, AnthropicTokenCounterAdapter)

    def test_returns_anthropic_adapter_for_claude_string(self) -> None:
        counter = create_token_counter("claude-3-opus-20240229")
        assert isinstance(counter, AnthropicTokenCounterAdapter)

    def test_returns_gemini_adapter_for_gemini_model(self) -> None:
        counter = create_token_counter(Model.GEMINI_2_5_PRO)
        assert isinstance(counter, GeminiTokenCounterAdapter)

    def test_returns_gemini_adapter_for_gemini_string(self) -> None:
        counter = create_token_counter("gemini-2.0-flash")
        assert isinstance(counter, GeminiTokenCounterAdapter)

    def test_returns_openai_adapter_for_unknown_model(self) -> None:
        counter = create_token_counter("unknown-model")
        assert isinstance(counter, OpenAITokenCounterAdapter)

    def test_returns_character_estimator_with_chars_per_token(self) -> None:
        chars_per_token = 4
        counter = create_token_counter("gpt-4o", chars_per_token=chars_per_token)
        assert isinstance(counter, CharacterEstimatorAdapter)
        assert counter.chars_per_token == chars_per_token

    def test_returns_openai_adapter_for_azure_openai(self) -> None:
        counter = create_token_counter("azure/gpt-4o")
        assert isinstance(counter, OpenAITokenCounterAdapter)


class TestProtocolConformance:
    def test_openai_adapter_conforms_to_protocol(self) -> None:
        counter = OpenAITokenCounterAdapter("gpt-4o")
        assert isinstance(counter, TokenCounterAdapter)

    def test_anthropic_adapter_conforms_to_protocol(self) -> None:
        counter = AnthropicTokenCounterAdapter("claude-3-opus-20240229")
        assert isinstance(counter, TokenCounterAdapter)

    def test_gemini_adapter_conforms_to_protocol(self) -> None:
        counter = GeminiTokenCounterAdapter("gemini-2.0-flash")
        assert isinstance(counter, TokenCounterAdapter)

    def test_character_estimator_conforms_to_protocol(self) -> None:
        estimator = CharacterEstimatorAdapter("any-model")
        assert isinstance(estimator, TokenCounterAdapter)
