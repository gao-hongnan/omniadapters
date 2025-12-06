from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model
    from pydantic_ai.providers import Provider

_CLIENT_KWARGS = frozenset({"api_key", "base_url", "http_client"})


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    model: str

    def create_agent(self, **overrides: Any) -> Agent[None, Any]:
        from pydantic_ai import Agent

        extras = dict(self.__pydantic_extra__ or {})
        extras.update(overrides)

        client_kwargs = {k: extras.pop(k) for k in list(extras) if k in _CLIENT_KWARGS}
        model_obj: str | Model = self._build_model(**client_kwargs) if client_kwargs else self.model

        kwargs: dict[str, Any] = {"model": model_obj}
        kwargs.update(extras)
        return Agent(**kwargs)

    def _build_model(self, **client_kwargs: Any) -> Model:
        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider_class

        def custom_provider_factory(name: str) -> Provider[Any]:
            provider_class = infer_provider_class(name)
            return provider_class(**client_kwargs)

        return infer_model(self.model, provider_factory=custom_provider_factory)


def create_agent(model: str, **kwargs: Any) -> Agent[None, Any]:
    from pydantic_ai import Agent

    return Agent(model=model, **kwargs)
