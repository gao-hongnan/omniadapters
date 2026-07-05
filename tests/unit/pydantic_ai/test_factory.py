"""Drift sentinels for the pydantic-ai ``KnownModelName`` grammar.

:mod:`omniadapters.pydantic_ai.factory` qualifies bare model names as
``f"{provider}:{model}"`` and hands the result to :func:`infer_model`. That only
works while pydantic-ai keeps its ``KnownModelName`` literal in the
``provider:model`` grammar (and keeps the ``test`` sentinel). These tests fail
loudly if upstream changes that grammar, so we catch the drift here rather than
at a customer call site.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic_ai.models import KnownModelName

from omniadapters.pydantic_ai.factory import _qualify_model_name

_KNOWN_MODEL_NAMES: frozenset[str] = frozenset(get_args(KnownModelName.__value__))

_MIN_PROVIDER_PREFIXED_LITERALS = 100


@pytest.mark.unit
class TestKnownModelNamesDrift:
    def test_resolves_to_non_empty_literal_set(self) -> None:
        assert _KNOWN_MODEL_NAMES, (
            "pydantic_ai.models.KnownModelName resolved to zero literal members. "
            "Upstream `TypeAliasType.__value__` access pattern may have changed; "
            "review `get_args(KnownModelName.__value__)`."
        )

    def test_includes_test_sentinel(self) -> None:
        assert "test" in _KNOWN_MODEL_NAMES, (
            "The 'test' sentinel literal is missing from pydantic_ai.KnownModelName. "
            "The factory relies on it to route TestModel without a provider prefix."
        )

    def test_includes_known_long_lived_anthropic_literal(self) -> None:
        assert "anthropic:claude-3-haiku-20240307" in _KNOWN_MODEL_NAMES, (
            "Long-stable literal 'anthropic:claude-3-haiku-20240307' missing from "
            "pydantic_ai.KnownModelName. Upstream may have removed it or changed the literal grammar."
        )

    def test_provider_prefix_grammar_is_dominant(self) -> None:
        prefixed = sum(1 for name in _KNOWN_MODEL_NAMES if ":" in name)
        assert prefixed >= _MIN_PROVIDER_PREFIXED_LITERALS, (
            f"Expected the dominant grammar of pydantic_ai.KnownModelName to be 'provider:model'; "
            f"found only {prefixed} colon-separated literals out of {len(_KNOWN_MODEL_NAMES)} "
            f"(threshold: {_MIN_PROVIDER_PREFIXED_LITERALS}). "
            "The factory's bare-name branch (`f'{provider}:{model_name}'`) depends on this grammar."
        )


@pytest.mark.unit
class TestQualifyModelName:
    """The shared surface-aware qualifier reused by build_model / create_embedder /
    create_image_generation.

    A bare name maps to a per-surface OpenAI prefix (``openai-chat:`` for text,
    ``openai:`` for embeddings, ``openai-responses:`` for image); non-OpenAI providers
    use ``<provider>:<name>``; a pre-qualified name (contains ``:``) is forwarded
    unchanged on every surface.
    """

    def test_qualify_pre_qualified_passthrough(self) -> None:
        for surface in ("text", "embeddings", "image"):
            assert (
                _qualify_model_name(provider_name="openai", model_name="openai:gpt-4o", surface=surface)
                == "openai:gpt-4o"
            )

    def test_qualify_non_openai_uses_provider_name(self) -> None:
        assert (
            _qualify_model_name(provider_name="google", model_name="gemini-2.5-flash", surface="text")
            == "google:gemini-2.5-flash"
        )

    def test_qualify_openai_text_prefix(self) -> None:
        assert _qualify_model_name(provider_name="openai", model_name="gpt-4o", surface="text") == "openai-chat:gpt-4o"

    def test_qualify_openai_embeddings_prefix(self) -> None:
        assert (
            _qualify_model_name(
                provider_name="openai",
                model_name="text-embedding-3-small",
                surface="embeddings",
            )
            == "openai:text-embedding-3-small"
        )

    def test_qualify_openai_image_prefix(self) -> None:
        assert (
            _qualify_model_name(provider_name="openai", model_name="gpt-5.4", surface="image")
            == "openai-responses:gpt-5.4"
        )
