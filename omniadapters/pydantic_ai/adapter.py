"""PydanticAI Adapter Module.

This module provides an adapter for creating PydanticAI Agent instances
with provider-specific configuration.

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
**Why Our TypeVars Are Invariant (No Variance)**

PydanticAI defines its Agent class with variance::

    AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
    OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)

Our TypeVars are intentionally **invariant** because they appear in
**both** input and output positions in our function signatures::

    def create_agent(
        deps_type: type[_DepsT],       # INPUT position
        output_type: type[_OutputT],   # INPUT position
    ) -> Agent[_DepsT, _OutputT]:      # OUTPUT position

Variance rules in type theory:

- **Covariant** (``covariant=True``): TypeVar can ONLY appear in output
  positions (return types, read-only attributes). Given ``Dog <: Animal``,
  ``Container[Dog] <: Container[Animal]``.

- **Contravariant** (``contravariant=True``): TypeVar can ONLY appear in
  input positions (parameters, write-only). Given ``Dog <: Animal``,
  ``Container[Animal] <: Container[Dog]``.

- **Invariant** (default): TypeVar can appear in both positions. No
  subtyping relationship between ``Container[Dog]`` and ``Container[Animal]``.

Since ``_DepsT`` appears as ``type[_DepsT]`` (input) AND in
``Agent[_DepsT, ...]`` (output), it must be invariant. Same for ``_OutputT``.

**The Returned Agent Still Has Proper Variance**

The ``Agent`` object we return follows PydanticAI's variance rules:

- ``Agent[Animal, str]`` is a subtype of ``Agent[Dog, str]`` (contravariant
  in deps)
- ``Agent[None, Dog]`` is a subtype of ``Agent[None, Animal]`` (covariant
  in output)

Our invariant TypeVars merely capture and propagate types; the variance on
``Agent``'s class-level type parameters governs the returned object's
subtyping behavior.

See Also
--------
pydantic_ai.Agent : The PydanticAI Agent class.
pydantic_ai._run_context.AgentDepsT : PydanticAI's contravariant deps TypeVar.
pydantic_ai.output.OutputDataT : PydanticAI's covariant output TypeVar.

"""

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any, get_args, overload

from pydantic_ai.models import KnownModelName, infer_model
from pydantic_ai.providers import infer_provider_class
from typing_extensions import TypeVar

_KNOWN_MODEL_NAMES: frozenset[str] = frozenset(get_args(KnownModelName))

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.output import OutputSpec
    from pydantic_ai.providers import Provider

    from ..core.models import ProviderConfig

_DepsT = TypeVar("_DepsT", default=None)
_OutputT = TypeVar("_OutputT", default=str)


class PydanticAIAdapter:
    def __init__(self, *, provider_config: ProviderConfig, model_name: KnownModelName | str) -> None:
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
    ) -> Agent[_DepsT, _OutputT]:
        from pydantic_ai import Agent  # NOTE: `Agent` can be a heavy import.

        provider_name: str = self.provider_config.provider
        client_kwargs = self.provider_config.get_client_kwargs()

        def custom_provider_factory(name: str) -> Provider[Any]:  # NOTE: pydantic-ai uses this type
            provider_class: type[Provider[Any]] = infer_provider_class(name)
            return provider_class(**client_kwargs)

        if self.model_name in _KNOWN_MODEL_NAMES:
            model = infer_model(self.model_name, provider_factory=custom_provider_factory)
        else:
            model_string = f"{provider_name}:{self.model_name}"
            model = infer_model(model_string, provider_factory=custom_provider_factory)

        final_kwargs: dict[str, Any] = {"model": model}
        if deps_type is not NoneType:
            final_kwargs["deps_type"] = deps_type
        if output_type is not None:
            final_kwargs["output_type"] = output_type
        final_kwargs.update(agent_kwargs)

        return Agent(**final_kwargs)
