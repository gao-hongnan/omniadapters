from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omniadapters.core.models import Usage


@singledispatch
def to_usage(provider_usage: object) -> Usage:
    """Convert provider-specific usage metadata to unified Usage model.

    This function uses Python's single-dispatch generic function pattern to
    automatically route conversion based on the runtime type of the input.
    Each LLM provider (OpenAI, Anthropic, Gemini, etc.) has different field
    names for token usage. This dispatcher centralizes the conversion logic
    while keeping implementations decoupled in their respective adapter modules.

    Parameters
    ----------
    provider_usage : object
        Provider-specific usage metadata object. Supported types are registered
        via the ``@to_usage.register`` decorator in each adapter module.

    Returns
    -------
    Usage
        Unified usage model with standardized field names.

    Raises
    ------
    NotImplementedError
        If no converter is registered for the given type.

    How Singledispatch Works
    ------------------------
    ``singledispatch`` is a decorator from ``functools`` that transforms a
    function into a **single-dispatch generic function**. "Single dispatch"
    means the implementation is chosen based on the type of a SINGLE argument
    (the first positional argument).

    **Registration Pattern**::

        @singledispatch
        def process(arg):
            '''Base/fallback implementation.'''
            raise NotImplementedError()

        @process.register(int)
        def _(arg: int):
            '''Called when arg is int.'''
            return arg * 2

        @process.register(str)
        def _(arg: str):
            '''Called when arg is str.'''
            return arg.upper()

    **Runtime Behavior**::

        process(5)       # -> 10 (dispatches to int handler)
        process("hi")    # -> "HI" (dispatches to str handler)
        process(3.14)    # -> NotImplementedError (no float handler)

    **Why Singledispatch for Usage Conversion?**

    1. **Open/Closed Principle**: Add new providers by registering handlers
       without modifying existing code.
    2. **Decoupled Registration**: Each adapter registers its converter at
       module import time, avoiding circular imports.
    3. **Type Safety**: Python dispatches based on actual runtime type.
    4. **Single Interface**: One function name (``to_usage``) handles all
       provider types.

    **Lazy Registration Pattern**::

        # In omniadapters/completion/adapters/openai.py
        from openai.types import CompletionUsage as OpenAIUsage
        from omniadapters.core.usage_converter import to_usage

        @to_usage.register(OpenAIUsage)
        def _(usage: OpenAIUsage) -> Usage:
            return Usage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                ...
            )

    When ``openai.py`` is imported, the handler is registered. This means:

    - No central file needs to import all provider SDKs
    - Only import provider types in their respective adapters
    - Third-party adapters can register their own handlers

    Examples
    --------
    Basic usage in an adapter:

    >>> from omniadapters.core.usage_converter import to_usage
    >>> # After adapter module registers its handler:
    >>> usage = to_usage(response.usage)  # Automatically dispatches

    Registering a custom provider:

    >>> from omniadapters.core.usage_converter import to_usage
    >>> from my_provider import MyUsageType
    >>>
    >>> @to_usage.register(MyUsageType)
    ... def _(usage: MyUsageType) -> Usage:
    ...     return Usage(
    ...         input_tokens=usage.tokens_in,
    ...         output_tokens=usage.tokens_out,
    ...         total_tokens=usage.tokens_in + usage.tokens_out,
    ...     )

    See Also
    --------
    functools.singledispatch : Python documentation for singledispatch.
    omniadapters.core.models.Usage : The unified usage model.

    """
    type_name = type(provider_usage).__name__
    msg = (
        f"No usage converter registered for type: {type_name}. Register a handler with @to_usage.register({type_name})"
    )
    raise NotImplementedError(msg)
