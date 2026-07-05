"""``create_embedder`` — the pydantic-ai embeddings factory for omniadapters.

Bridges an omniadapters :class:`ProviderConfig` to a native pydantic-ai
:class:`Embedder`. The provider is built once (carrying the config's ``api_key``)
and injected via ``provider_factory`` into :func:`infer_embedding_model`, because
``Embedder.__init__`` has no ``provider_factory`` and would otherwise re-infer the
provider from the environment. A bare OpenAI model name is qualified ``openai:``
(the embeddings endpoint has no chat-vs-responses split), never ``openai-chat:``.

The returned ``Embedder`` is native pydantic-ai: its ``embed_query`` /
``embed_documents`` yield ``EmbeddingResult``, and omniadapters neither wraps nor
normalizes it. A provider without an embeddings endpoint surfaces a pydantic-ai
``ModelHTTPError`` at embed time (EC-005), not at construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.embeddings import (
    Embedder,
    EmbeddingResult,
    EmbeddingSettings,
    infer_embedding_model,
)

from .factory import _qualify_model_name, build_provider

if TYPE_CHECKING:
    from ..core.models import ProviderConfig

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "EmbeddingSettings",
    "create_embedder",
]


def create_embedder(
    *,
    provider_config: ProviderConfig,
    model_name: str,
    settings: EmbeddingSettings | None = None,
) -> Embedder:
    """Create a native pydantic-ai ``Embedder`` wired to an omniadapters config.

    Builds the provider from ``provider_config`` (carrying its ``api_key``),
    qualifies ``model_name`` for the embeddings surface (bare OpenAI →
    ``"openai:<name>"``), and resolves it through :func:`infer_embedding_model`
    with a ``provider_factory`` that returns the pre-built provider, so the
    api_key is injected rather than re-inferred from the environment. A
    pre-qualified ``model_name`` (already containing ``":"``) is forwarded
    unchanged. Returns the native ``Embedder`` — no omniadapters wrapper type.
    """
    provider = build_provider(provider_config)

    def factory(_: str) -> Any:
        return provider

    qualified = _qualify_model_name(provider_name=provider.name, model_name=model_name, surface="embeddings")
    model = infer_embedding_model(qualified, provider_factory=factory)
    return Embedder(model=model, settings=settings)
