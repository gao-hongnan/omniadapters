from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

_BUNDLED_CACHE_DIR = Path(__file__).parent.parent / ".tiktoken_cache"
if "TIKTOKEN_CACHE_DIR" not in os.environ and _BUNDLED_CACHE_DIR.exists():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_BUNDLED_CACHE_DIR)

import tiktoken

from omniadapters.core.constants import ANTHROPIC_IMPORT_ERROR, GEMINI_IMPORT_ERROR
from omniadapters.core.enums import Model, Provider, infer_provider

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from google import genai

FALLBACK_ENCODING = "cl100k_base"


@runtime_checkable
class TokenCounterAdapter(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def provider(self) -> Provider | None: ...

    def count_tokens(self, text: str) -> int: ...

    async def acount_tokens(self, text: str) -> int: ...


class BaseTokenCounterAdapter(ABC):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def provider(self) -> Provider | None: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    async def acount_tokens(self, text: str) -> int: ...


class OpenAITokenCounterAdapter(BaseTokenCounterAdapter):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._encoding = _get_encoding(model)

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    async def acount_tokens(self, text: str) -> int:
        return self.count_tokens(text)


class AnthropicTokenCounterAdapter(BaseTokenCounterAdapter):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        self._api_key = api_key
        self._client: AsyncAnthropic | None = None
        self._fallback_encoding = tiktoken.get_encoding(FALLBACK_ENCODING)

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    def count_tokens(self, text: str) -> int:
        return len(self._fallback_encoding.encode(text))

    async def acount_tokens(self, text: str) -> int:
        if self._api_key is None:
            return self.count_tokens(text)

        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(ANTHROPIC_IMPORT_ERROR) from e
            self._client = AsyncAnthropic(api_key=self._api_key)

        response = await self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens


class GeminiTokenCounterAdapter(BaseTokenCounterAdapter):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        self._api_key = api_key
        self._client: genai.Client | None = None
        self._fallback_encoding = tiktoken.get_encoding(FALLBACK_ENCODING)

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    def count_tokens(self, text: str) -> int:
        return len(self._fallback_encoding.encode(text))

    async def acount_tokens(self, text: str) -> int:
        if self._api_key is None:
            return self.count_tokens(text)

        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(GEMINI_IMPORT_ERROR) from e
            self._client = genai.Client(api_key=self._api_key)

        response = await self._client.aio.models.count_tokens(
            model=self._model,
            contents=text,
        )
        return response.total_tokens or 0


class CharacterEstimatorAdapter(BaseTokenCounterAdapter):
    def __init__(self, model: str, chars_per_token: int = 4) -> None:
        super().__init__(model)
        self._chars_per_token = chars_per_token

    @property
    def provider(self) -> Provider | None:
        return None

    @property
    def chars_per_token(self) -> int:
        return self._chars_per_token

    def count_tokens(self, text: str) -> int:
        return len(text) // self._chars_per_token

    async def acount_tokens(self, text: str) -> int:
        return self.count_tokens(text)


type TokenCounterAdapterType = (
    OpenAITokenCounterAdapter | AnthropicTokenCounterAdapter | GeminiTokenCounterAdapter | CharacterEstimatorAdapter
)


def create_token_counter(
    model: Model | str,
    *,
    api_key: str | None = None,
    chars_per_token: int | None = None,
) -> TokenCounterAdapterType:
    if chars_per_token is not None:
        # Fast path for character estimator, source OpenAI: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        model_str = model.value if isinstance(model, Model) else model
        return CharacterEstimatorAdapter(model=model_str, chars_per_token=chars_per_token)

    provider = infer_provider(model)
    model_str = model.value if isinstance(model, Model) else model

    match provider:
        case Provider.OPENAI | Provider.AZURE_OPENAI:
            return OpenAITokenCounterAdapter(model=model_str)
        case Provider.ANTHROPIC:
            return AnthropicTokenCounterAdapter(model=model_str, api_key=api_key)
        case Provider.GEMINI:
            return GeminiTokenCounterAdapter(model=model_str, api_key=api_key)
        case _:
            return OpenAITokenCounterAdapter(model=model_str)


@lru_cache(maxsize=16)
def _get_encoding(model: Model | str) -> tiktoken.Encoding:
    model_str = model.value if isinstance(model, Model) else model
    try:
        return tiktoken.encoding_for_model(model_str)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)
