"""Unit tests for ``create_image_generation`` and the vision-input re-exports.

The factory forwards image settings opaquely to the native
:class:`pydantic_ai.capabilities.ImageGeneration` and, when both fallback args
are supplied, builds the fallback ``Model`` via the shared ``build_provider`` +
``_qualify_model_name(surface="image")`` + ``infer_model`` primitives — leaving
``build_model``'s signature untouched (SC-002). ``infer_model`` is spied on so no
network is touched; live image generation is deferred to manual/integration
(SC-004). The patch seam is ``infer_model`` as imported into
``omniadapters.pydantic_ai.images``.
"""

from __future__ import annotations

from unittest.mock import patch

import pydantic_ai
import pytest
from pydantic import SecretStr
from pydantic_ai.capabilities import ImageGeneration

from omniadapters.core.models import OpenAIProviderConfig
from omniadapters.pydantic_ai import images
from omniadapters.pydantic_ai.images import create_image_generation


@pytest.mark.unit
class TestCreateImageGeneration:
    def test_create_image_generation_returns_capability(self) -> None:
        with patch("omniadapters.pydantic_ai.images.infer_model"):
            cap = create_image_generation(
                fallback_provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
                fallback_model_name="gpt-5.4",
            )
        assert isinstance(cap, ImageGeneration)

    def test_create_image_generation_fallback_uses_responses_prefix(self) -> None:
        with patch("omniadapters.pydantic_ai.images.infer_model") as mock_infer:
            create_image_generation(
                fallback_provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
                fallback_model_name="gpt-5.4",
            )
        assert mock_infer.call_args.args[0] == "openai-responses:gpt-5.4"

    def test_create_image_generation_forwards_image_kwargs(self) -> None:
        cap = create_image_generation(quality="high", size="1024x1024")
        assert cap.quality == "high"
        assert cap.size == "1024x1024"

    def test_create_image_generation_no_fallback_omits_fallback_model(self) -> None:
        cap = create_image_generation()
        assert isinstance(cap, ImageGeneration)
        assert cap.fallback_model is None

    def test_vision_parts_reexported(self) -> None:
        assert images.BinaryContent is pydantic_ai.BinaryContent
        assert images.BinaryImage is pydantic_ai.BinaryImage
        assert images.ImageUrl is pydantic_ai.ImageUrl
