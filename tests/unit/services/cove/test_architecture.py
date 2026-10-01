"""Import boundaries: the engine core stays provider-agnostic; models enter only through the pydantic-ai backend."""

from __future__ import annotations

import subprocess
import sys
from typing import Final

import pytest

_PROVIDER_SDKS: Final = ("instructor", "openai", "anthropic", "google.genai", "pydantic_ai")
_BACKEND_AND_RECIPES: Final = (
    "import omniadapters.services.cove.backends, omniadapters.services.cove.recipes.factual_qa"
)


def _modules_loaded_by(statement: str, candidates: tuple[str, ...] = _PROVIDER_SDKS) -> list[str]:
    probe = f"import sys\n{statement}\nprint(','.join(name for name in {candidates!r} if name in sys.modules))"
    completed = subprocess.run(  # noqa: S603 - fixed argv on the running interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return [name for name in completed.stdout.strip().split(",") if name]


@pytest.mark.unit
class TestImportBoundaries:
    def test_the_engine_core_loads_no_provider_sdk(self) -> None:
        assert _modules_loaded_by("import omniadapters.services.cove") == []

    def test_models_come_in_through_the_pydantic_ai_backend(self) -> None:
        assert "pydantic_ai" in _modules_loaded_by("import omniadapters.services.cove.backends")

    def test_no_role_goes_through_structify(self) -> None:
        assert _modules_loaded_by(_BACKEND_AND_RECIPES, candidates=("omniadapters.structify",)) == []
