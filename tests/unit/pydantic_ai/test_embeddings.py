"""Unit tests for ``create_embedder`` — the pydantic-ai embeddings factory.

These assert the config-bridge wiring — provider construction, surface-aware
qualification (``openai:`` not ``openai-chat:``), and pre-built-provider
injection — without touching the network: provider construction is offline and
:func:`infer_embedding_model` is spied on. Live embedding calls are deferred to
manual/integration (SC-003).

The patch seam is ``infer_embedding_model`` / ``build_provider`` as imported into
``omniadapters.pydantic_ai.embeddings`` (not the pydantic-ai origin), mirroring
the mocking precedent in ``tests/unit/completion/test_map_api_errors.py``.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr
from pydantic_ai.embeddings import Embedder

from omniadapters.core.models import OpenAIProviderConfig
from omniadapters.pydantic_ai.embeddings import create_embedder


@pytest.mark.unit
class TestCreateEmbedder:
    def test_create_embedder_returns_native_embedder(self) -> None:
        embedder = create_embedder(
            provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
            model_name="text-embedding-3-small",
        )
        assert isinstance(embedder, Embedder)
        assert inspect.iscoroutinefunction(embedder.embed_query)
        assert inspect.iscoroutinefunction(embedder.embed_documents)

    def test_create_embedder_uses_openai_embeddings_prefix(self) -> None:
        with patch(
            "omniadapters.pydantic_ai.embeddings.infer_embedding_model"
        ) as mock_infer:
            create_embedder(
                provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
                model_name="text-embedding-3-small",
            )
        mock_infer.assert_called_once()
        assert mock_infer.call_args.args[0] == "openai:text-embedding-3-small"

    def test_create_embedder_injects_prebuilt_provider(self) -> None:
        sentinel = MagicMock()
        sentinel.name = "openai"
        with (
            patch(
                "omniadapters.pydantic_ai.embeddings.build_provider",
                return_value=sentinel,
            ),
            patch(
                "omniadapters.pydantic_ai.embeddings.infer_embedding_model"
            ) as mock_infer,
        ):
            create_embedder(
                provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
                model_name="text-embedding-3-small",
            )
        provider_factory = mock_infer.call_args.kwargs["provider_factory"]
        assert provider_factory("any-provider-name") is sentinel

    def test_create_embedder_pre_qualified_name_forwarded(self) -> None:
        with patch(
            "omniadapters.pydantic_ai.embeddings.infer_embedding_model"
        ) as mock_infer:
            create_embedder(
                provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
                model_name="openai:text-embedding-3-large",
            )
        assert mock_infer.call_args.args[0] == "openai:text-embedding-3-large"
