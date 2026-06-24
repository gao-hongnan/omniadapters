"""Factory functions for creating pydantic-ai providers, models and agents.

This is the single source of truth for the omniadapters → pydantic-ai bridge:

- ``build_provider`` turns a ``ProviderConfig`` into a pydantic-ai ``Provider``.
- ``build_model`` resolves a model name against a ``ProviderConfig``.
- ``create_agent`` builds a fully-configured ``Agent`` with overloads that
  propagate ``deps_type`` / ``output_type`` exactly like ``Agent(...)``.

Our ``Provider`` enum values already follow pydantic-ai's naming, so no name
translation is required when building the provider. The only qualification
concern is the OpenAI prefix: pydantic-ai resolves a bare ``openai:`` prefix to
the Responses API, so a bare model name is qualified ``openai-chat:`` to keep
the Chat Completions behaviour this factory has always produced. A
caller-supplied prefix (``"openai:gpt-4o"``, ``"google:gemini-2.5-flash"``) is
forwarded as-is.
"""

from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, Any, overload

from pydantic_ai import Agent
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider_class

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.models import Model
    from pydantic_ai.output import OutputSpec
    from pydantic_ai.providers import Provider
    from pydantic_ai.tools import Tool, ToolFuncEither

    from ..core.models import ProviderConfig


def build_provider(provider_config: ProviderConfig) -> Provider[Any]:
    """Build a pydantic-ai ``Provider`` from an omniadapters ``ProviderConfig``.

    ``provider_config.provider`` already follows pydantic-ai's naming, so it is
    resolved straight to a provider class and instantiated with the config's
    client kwargs (``api_key`` plus any provider-specific extras).
    """
    provider_class = infer_provider_class(provider_config.provider)
    return provider_class(**provider_config.get_client_kwargs())


def build_model(*, provider_config: ProviderConfig, model_name: str) -> Model:
    """Resolve ``model_name`` against ``provider_config``'s provider.

    A bare ``model_name`` (no ``"<prefix>:"``) is qualified with the built
    provider's name — e.g. ``"google:gemini-2.5-flash"`` — except OpenAI, which
    is qualified ``"openai-chat:"`` so it stays on the Chat Completions API (a
    bare ``"openai:"`` resolves to the Responses API). A pre-qualified
    ``model_name`` is forwarded unchanged, letting a caller pass
    ``"openai:gpt-4o"`` to opt into the Responses API.
    """
    provider = build_provider(provider_config)

    def factory(_: str) -> Any:
        return provider

    if ":" in model_name:
        qualified = model_name
    else:
        prefix = "openai-chat" if provider.name == "openai" else provider.name
        qualified = f"{prefix}:{model_name}"

    return infer_model(qualified, provider_factory=factory)


# ``output_type`` mirrors pydantic-ai's own ``OutputSpec[OutputT]`` (not just
# ``type[OutputT]``), so the full output surface — a type, an output function, a
# ``ToolOutput`` / ``NativeOutput`` / ``PromptedOutput`` / ``TextOutput`` marker,
# or a union of those — type-checks through this factory exactly as it does
# through ``Agent(...)`` directly. Ordering matters: the deps-*required*
# overloads must precede the deps-*defaulted* ones, otherwise mypy reports an
# ``[overload-overlap]`` between the two ``OutputSpec`` variants.
@overload
def create_agent[DepsT, OutputT](
    *,
    provider_config: ProviderConfig,
    model_name: str,
    output_type: OutputSpec[OutputT],
    deps_type: type[DepsT],
    tools: Sequence[ToolFuncEither[DepsT, ...] | Tool[DepsT]] | None = ...,
    **agent_kwargs: Any,
) -> Agent[DepsT, OutputT]: ...
@overload
def create_agent[DepsT](
    *,
    provider_config: ProviderConfig,
    model_name: str,
    output_type: None = ...,
    deps_type: type[DepsT],
    tools: Sequence[ToolFuncEither[DepsT, ...] | Tool[DepsT]] | None = ...,
    **agent_kwargs: Any,
) -> Agent[DepsT, str]: ...
@overload
def create_agent[OutputT](
    *,
    provider_config: ProviderConfig,
    model_name: str,
    output_type: OutputSpec[OutputT],
    deps_type: type[None] = ...,
    tools: Sequence[ToolFuncEither[None, ...] | Tool[None]] | None = ...,
    **agent_kwargs: Any,
) -> Agent[None, OutputT]: ...
@overload
def create_agent(
    *,
    provider_config: ProviderConfig,
    model_name: str,
    output_type: None = ...,
    deps_type: type[None] = ...,
    tools: Sequence[ToolFuncEither[None, ...] | Tool[None]] | None = ...,
    **agent_kwargs: Any,
) -> Agent[None, str]: ...
def create_agent(
    *,
    provider_config: ProviderConfig,
    model_name: str,
    output_type: OutputSpec[Any] | None = None,
    deps_type: type[Any] = NoneType,
    tools: Sequence[ToolFuncEither[Any, ...] | Tool[Any]] | None = None,
    **agent_kwargs: Any,
) -> Agent[Any, Any]:
    """Create a pydantic-ai ``Agent`` wired to an omniadapters provider config.

    Builds the model from ``provider_config`` / ``model_name`` (see
    :func:`build_model`) and forwards ``output_type``, ``deps_type``, ``tools``
    and any extra ``agent_kwargs`` straight to ``Agent(...)``.
    """
    model = build_model(provider_config=provider_config, model_name=model_name)
    kwargs: dict[str, Any] = {"model": model, "deps_type": deps_type}
    if output_type is not None:
        kwargs["output_type"] = output_type
    if tools:
        kwargs["tools"] = list(tools)
    kwargs.update(agent_kwargs)
    return Agent(**kwargs)
