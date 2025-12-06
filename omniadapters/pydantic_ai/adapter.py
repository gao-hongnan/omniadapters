from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from ..core.types import ProviderConfigT

if TYPE_CHECKING:
    from pydantic_ai import Agent


class PydanticAIAdapter(Generic[ProviderConfigT]):
    def __init__(self, *, provider_config: ProviderConfigT, model_name: str) -> None:
        self.provider_config = provider_config
        self.model_name = model_name

    def create_agent(self, **agent_kwargs: Any) -> Agent[None, Any]:
        from pydantic_ai import Agent
        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider_class

        provider_name: str = self.provider_config.provider
        client_kwargs = self.provider_config.get_client_kwargs()

        def custom_provider_factory(name: str) -> Any:
            provider_class = infer_provider_class(name)
            return provider_class(**client_kwargs)

        model_string = f"{provider_name}:{self.model_name}"
        model = infer_model(model_string, provider_factory=custom_provider_factory)

        return Agent(model=model, **agent_kwargs)
