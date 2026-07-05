"""PydanticAI integration for omniadapters.

Re-exports the pydantic-ai types most agent code needs, plus three factory
functions that wire omniadapters :class:`ProviderConfig` into pydantic-ai:

``build_provider``: build a pydantic-ai ``Provider`` from a ``ProviderConfig``.
``build_model``: build a pydantic-ai ``Model`` from a config + model name.
``create_agent``: build a fully-configured ``Agent`` (typed overloads).
"""

from __future__ import annotations

from ..core.constants import PYDANTIC_AI_IMPORT_ERROR

try:
    from genai_prices.types import PriceCalculation
    from pydantic_ai import (
        Agent,
        BinaryContent,
        BinaryImage,
        ImageUrl,
        ModelRetry,
        RunContext,
        Tool,
    )
    from pydantic_ai.capabilities import ImageGeneration
    from pydantic_ai.embeddings import Embedder, EmbeddingResult
    from pydantic_ai.result import FinalResult, StreamedRunResult
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
except ImportError as e:
    raise ImportError(PYDANTIC_AI_IMPORT_ERROR) from e

from ..core.cost import CostAccumulator, Unpriced, compute_cost
from .embeddings import create_embedder
from .factory import build_model, build_provider, create_agent
from .images import create_image_generation

__all__ = [
    # Core types from pydantic-ai
    "Agent",
    "BinaryContent",
    "BinaryImage",
    "CostAccumulator",
    "Embedder",
    "EmbeddingResult",
    "FinalResult",
    "ImageGeneration",
    "ImageUrl",
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
    # omniadapters pydantic_ai factory functions
    "build_model",
    "build_provider",
    "compute_cost",
    "create_agent",
    "create_embedder",
    "create_image_generation",
]
