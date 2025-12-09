"""PydanticAI Agent Configuration Module.

This module provides a Pydantic-based configuration model for creating
PydanticAI Agent instances with validation and serialization support.

Type Parameters
---------------
_DepsT : TypeVar, default=None
    Type variable for agent dependencies. Captures the dependency type
    from ``deps_type`` parameter and propagates it to the returned
    ``Agent[_DepsT, _OutputT]``.

_OutputT : TypeVar, default=str
    Type variable for agent output. Captures the output type from
    ``output_type`` parameter and propagates it to the returned
    ``Agent[_DepsT, _OutputT]``.

Notes
-----
Our TypeVars are **invariant** (no variance) because they appear in both
input and output positions. See ``adapter.py`` module docstring for the
full explanation of variance rules and why invariance is required here.

See Also
--------
omniadapters.pydantic_ai.adapter : Detailed variance explanation.
pydantic_ai.Agent : The PydanticAI Agent class.

"""

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model
    from pydantic_ai.output import OutputSpec
    from pydantic_ai.providers import Provider

_CLIENT_KWARGS = frozenset({"api_key", "base_url", "http_client"})

_DepsT = TypeVar("_DepsT", default=None)
_OutputT = TypeVar("_OutputT", default=str)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    model: str

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[None] = ...,
        output_type: None = ...,
        **overrides: Any,
    ) -> Agent[None, str]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[None] = ...,
        output_type: type[_OutputT],
        **overrides: Any,
    ) -> Agent[None, _OutputT]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[_DepsT],
        output_type: None = ...,
        **overrides: Any,
    ) -> Agent[_DepsT, str]: ...

    @overload
    def create_agent(
        self,
        *,
        deps_type: type[_DepsT],
        output_type: type[_OutputT],
        **overrides: Any,
    ) -> Agent[_DepsT, _OutputT]: ...

    def create_agent(
        self,
        *,
        deps_type: type[_DepsT | None] = NoneType,
        output_type: OutputSpec[_OutputT] | None = None,
        **overrides: Any,
    ) -> Agent[_DepsT, _OutputT]:
        from pydantic_ai import Agent

        extras = dict(self.__pydantic_extra__ or {})
        extras.update(overrides)

        client_kwargs = {k: extras.pop(k) for k in list(extras) if k in _CLIENT_KWARGS}
        model_obj: str | Model = self._build_model(**client_kwargs) if client_kwargs else self.model

        kwargs: dict[str, Any] = {"model": model_obj}
        if deps_type is not NoneType:
            kwargs["deps_type"] = deps_type
        if output_type is not None:
            kwargs["output_type"] = output_type
        kwargs.update(extras)
        return Agent(**kwargs)

    def _build_model(self, **client_kwargs: Any) -> Model:
        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider_class

        def custom_provider_factory(name: str) -> Provider[object]:
            provider_class = infer_provider_class(name)
            return provider_class(**client_kwargs)

        return infer_model(self.model, provider_factory=custom_provider_factory)
