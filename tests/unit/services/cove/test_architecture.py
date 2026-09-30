"""Import boundaries: the engine core stays provider-agnostic; providers enter only through ``backends``."""

from __future__ import annotations

import subprocess
import sys
from typing import Final

import pytest

_PROVIDER_SDKS: Final = ("instructor", "openai", "anthropic", "google.genai", "pydantic_ai")


def _modules_loaded_by(statement: str) -> list[str]:
    probe = f"import sys\n{statement}\nprint(','.join(name for name in {_PROVIDER_SDKS!r} if name in sys.modules))"
    completed = subprocess.run(  # noqa: S603 - fixed argv on the running interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return [name for name in completed.stdout.strip().split(",") if name]


@pytest.mark.unit
class TestImportBoundaries:
    def test_the_engine_core_loads_no_provider_sdk(self) -> None:
        assert _modules_loaded_by("import omniadapters.services.cove") == []

    def test_providers_come_in_through_backends(self) -> None:
        assert "instructor" in _modules_loaded_by("import omniadapters.services.cove.backends")
