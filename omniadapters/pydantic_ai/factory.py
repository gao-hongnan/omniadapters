from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter import PydanticAIAdapter

if TYPE_CHECKING:
    from ..core.models import ProviderConfig


def create_adapter(*, provider_config: ProviderConfig, model_name: str) -> PydanticAIAdapter[ProviderConfig]:
    return PydanticAIAdapter(provider_config=provider_config, model_name=model_name)
