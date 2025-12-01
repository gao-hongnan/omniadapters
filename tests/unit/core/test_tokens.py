from __future__ import annotations

from omniadapters.core.enums import Model
from omniadapters.core.tokens import CharacterEstimator, TikTokenCounter, TokenCounter, get_token_counter


class TestTikTokenCounter:
    def test_init_with_model_enum(self) -> None:
        counter = TikTokenCounter(Model.GPT_4O)
        assert counter.model == Model.GPT_4O

    def test_init_with_model_string(self) -> None:
        counter = TikTokenCounter("gpt-4o")
        assert counter.model == "gpt-4o"

    def test_count_tokens(self) -> None:
        counter = TikTokenCounter()
        count = counter.count_tokens("Hello world")
        assert count > 0

    def test_count_tokens_consistency(self) -> None:
        counter = TikTokenCounter()
        text = "This is a test sentence for token counting."
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        assert count1 == count2

    def test_count_tokens_unknown_model_fallback(self) -> None:
        counter = TikTokenCounter("unknown-model")
        count = counter.count_tokens("Hello world")
        assert count > 0


class TestCharacterEstimator:
    DEFAULT_CHARS_PER_TOKEN = 4
    CUSTOM_CHARS_PER_TOKEN = 3

    def test_default_chars_per_token(self) -> None:
        estimator = CharacterEstimator()
        assert estimator.chars_per_token == self.DEFAULT_CHARS_PER_TOKEN

    def test_custom_chars_per_token(self) -> None:
        estimator = CharacterEstimator(chars_per_token=self.CUSTOM_CHARS_PER_TOKEN)
        assert estimator.chars_per_token == self.CUSTOM_CHARS_PER_TOKEN

    def test_count_tokens(self) -> None:
        estimator = CharacterEstimator(chars_per_token=self.DEFAULT_CHARS_PER_TOKEN)
        count = estimator.count_tokens("12345678")
        expected_tokens = 8 // self.DEFAULT_CHARS_PER_TOKEN
        assert count == expected_tokens


class TestGetTokenCounter:
    def test_returns_tiktoken_counter(self) -> None:
        counter = get_token_counter(Model.GPT_4O)
        assert isinstance(counter, TikTokenCounter)

    def test_with_string_model(self) -> None:
        counter = get_token_counter("gpt-4o")
        assert isinstance(counter, TikTokenCounter)


class TestProtocolConformance:
    def test_tiktoken_counter_conforms_to_token_counter(self) -> None:
        counter = TikTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_character_estimator_conforms_to_token_counter(self) -> None:
        estimator = CharacterEstimator()
        assert isinstance(estimator, TokenCounter)
