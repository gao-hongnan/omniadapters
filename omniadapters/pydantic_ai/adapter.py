from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.output import OutputSpec

    from ..core.models import ProviderConfig

_DepsT = TypeVar("_DepsT")
_OutputT = TypeVar("_OutputT")


class PydanticAIAdapter:
    def __init__(self, *, provider_config: ProviderConfig, model_name: str) -> None:
        self.provider_config = provider_config
        self.model_name = model_name

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[None] = ...,
        output_type: None = ...,
        **agent_kwargs: Any,
    ) -> Agent[None, str]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[None] = ...,
        output_type: type[_OutputT],
        **agent_kwargs: Any,
    ) -> Agent[None, _OutputT]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[_DepsT],
        output_type: None = ...,
        **agent_kwargs: Any,
    ) -> Agent[_DepsT, str]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[_DepsT],
        output_type: type[_OutputT],
        **agent_kwargs: Any,
    ) -> Agent[_DepsT, _OutputT]: ...

    def create_agent(
        self,
        *,
        deps_type: type[_DepsT | None] = NoneType,
        output_type: OutputSpec[_OutputT] | None = None,
        **agent_kwargs: Any,
    ) -> Agent[Any, Any]:
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

        final_kwargs: dict[str, Any] = {"model": model}
        if deps_type is not NoneType:
            final_kwargs["deps_type"] = deps_type
        if output_type is not None:
            final_kwargs["output_type"] = output_type
        final_kwargs.update(agent_kwargs)

        return Agent(**final_kwargs)
