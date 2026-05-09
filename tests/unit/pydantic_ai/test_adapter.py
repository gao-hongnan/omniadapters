from __future__ import annotations

import pytest

from omniadapters.pydantic_ai.adapter import _KNOWN_MODEL_NAMES

_MIN_PROVIDER_PREFIXED_LITERALS = 100


@pytest.mark.unit
class TestKnownModelNamesDrift:
    def test_resolves_to_non_empty_literal_set(self) -> None:
        assert _KNOWN_MODEL_NAMES, (
            "pydantic_ai.models.KnownModelName resolved to zero literal members. "
            "Upstream `TypeAliasType.__value__` access pattern may have changed; "
            "review `get_args(KnownModelName.__value__)` in adapter.py."
        )

    def test_includes_test_sentinel(self) -> None:
        assert "test" in _KNOWN_MODEL_NAMES, (
            "The 'test' sentinel literal is missing from pydantic_ai.KnownModelName. "
            "The adapter relies on it to route TestModel without a provider prefix."
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
            "The adapter's else-branch (`f'{provider_name}:{self.model_name}'`) depends on this grammar."
        )
