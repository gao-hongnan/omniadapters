"""Public-surface tests for ``omniadapters.pydantic_ai``.

Assert the single-line import (FR-007) resolves the nine agent/embeddings/image/
vision names, that ``__all__`` is a strict superset of the prior 17 exports
(FR-010), and that ``ImageGeneration`` resolves to the capability class from
``pydantic_ai.capabilities`` rather than the top-level ``ImageGenerationTool``
(EC-006).
"""

from __future__ import annotations

import pydantic_ai
import pytest
from pydantic_ai.capabilities import ImageGeneration as CapabilityImageGeneration

import omniadapters.pydantic_ai as pai
from omniadapters.pydantic_ai import (
    BinaryContent,
    BinaryImage,
    Embedder,
    EmbeddingResult,
    ImageGeneration,
    ImageUrl,
    create_agent,
    create_embedder,
    create_image_generation,
)

_PRIOR_17: frozenset[str] = frozenset(
    {
        "Agent",
        "CostAccumulator",
        "FinalResult",
        "ModelRetry",
        "ModelSettings",
        "PriceCalculation",
        "RequestUsage",
        "RunContext",
        "RunUsage",
        "StreamedRunResult",
        "Tool",
        "Unpriced",
        "UsageLimits",
        "build_model",
        "build_provider",
        "compute_cost",
        "create_agent",
    }
)

_NEW_8: frozenset[str] = frozenset(
    {
        "create_image_generation",
        "create_embedder",
        "ImageGeneration",
        "Embedder",
        "EmbeddingResult",
        "BinaryContent",
        "BinaryImage",
        "ImageUrl",
    }
)


@pytest.mark.unit
class TestPublicSurface:
    def test_single_line_import_resolves_all_nine(self) -> None:
        for obj in (
            create_agent,
            create_image_generation,
            create_embedder,
            ImageGeneration,
            Embedder,
            EmbeddingResult,
            BinaryContent,
            BinaryImage,
            ImageUrl,
        ):
            assert obj is not None

    def test_all_is_strict_superset_of_prior_17(self) -> None:
        exported = set(pai.__all__)
        assert _PRIOR_17 <= exported
        assert _NEW_8 <= exported

    def test_image_generation_is_capability_not_tool(self) -> None:
        assert ImageGeneration is CapabilityImageGeneration
        assert ImageGeneration is not pydantic_ai.ImageGenerationTool
