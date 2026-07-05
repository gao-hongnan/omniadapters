"""``create_image_generation`` plus vision-input re-exports for omniadapters.

Bridges an omniadapters :class:`ProviderConfig` to pydantic-ai's native image
capability. ``create_image_generation`` forwards image settings opaquely to
:class:`pydantic_ai.capabilities.ImageGeneration` (imported from
``pydantic_ai.capabilities`` — the top-level ``pydantic_ai`` exports only
``ImageGenerationTool``). When both fallback arguments are supplied, it builds the
fallback ``Model`` from the shared ``build_provider`` +
``_qualify_model_name(surface="image")`` + ``infer_model`` primitives, qualifying a
bare OpenAI name to ``openai-responses:`` (the Responses API). ``build_model``'s
signature is deliberately left unchanged (SC-002), so the fallback is resolved
here rather than through it.

No omniadapters image-only-model guard is added: pydantic-ai's own guard fires
only for string model names, so passing a built ``Model`` instance (the config
bridge requires the ``api_key``) intentionally bypasses it — a documented
tradeoff. The vision-input message parts ``BinaryContent`` / ``BinaryImage`` /
``ImageUrl`` are re-exported so a caller attaches images to an agent prompt with
one import; attach the returned capability via ``create_agent(..., capabilities=[cap])``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import BinaryContent, BinaryImage, ImageUrl
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.models import infer_model

from .factory import _qualify_model_name, build_provider

if TYPE_CHECKING:
    from ..core.models import ProviderConfig

__all__ = [
    "BinaryContent",
    "BinaryImage",
    "ImageGeneration",
    "ImageUrl",
    "create_image_generation",
]


def create_image_generation(
    *,
    fallback_provider_config: ProviderConfig | None = None,
    fallback_model_name: str | None = None,
    **image_kwargs: Any,
) -> ImageGeneration:
    """Create a native pydantic-ai ``ImageGeneration`` capability.

    ``image_kwargs`` (e.g. ``quality``, ``size``, ``output_format``) are forwarded
    opaquely to ``ImageGeneration(...)``. When both ``fallback_provider_config`` and
    ``fallback_model_name`` are supplied, a fallback ``Model`` is built and set as
    ``image_kwargs["fallback_model"]`` via ``setdefault`` (an explicit caller-passed
    ``fallback_model`` wins); a bare OpenAI fallback name is qualified
    ``openai-responses:``. Attach the result via ``create_agent(..., capabilities=[cap])``.
    """
    if fallback_provider_config is not None and fallback_model_name is not None:
        provider = build_provider(fallback_provider_config)

        def factory(_: str) -> Any:
            return provider

        qualified = _qualify_model_name(
            provider_name=provider.name,
            model_name=fallback_model_name,
            surface="image",
        )
        image_kwargs.setdefault("fallback_model", infer_model(qualified, provider_factory=factory))

    return ImageGeneration(**image_kwargs)
