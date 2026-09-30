"""The LLM port, the step that puts an LLM on the stand, and decorators for the port.

:class:`StructuredLLM` has one method, so cross-cutting concerns compose as decorators around it
instead of as engine features: :class:`Throttled` caps concurrent calls across roles, runs and
engines, and :class:`Budgeted` stops spending once a shared :class:`CallBudget` is gone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload

from pydantic import BaseModel

from .errors import BudgetExceededError, TemplateVariableError
from .prompts import JinjaPrompt

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from _typeshed import DataclassInstance

    from .enums import Role
    from .prompts import Messages, Prompt


class StructuredLLM(Protocol):
    """Chat messages in, a validated ``response_model`` instance out.

    Every ``omniadapters.structify`` adapter satisfies this protocol unchanged. So do
    :class:`~omniadapters.services.cove.backends.PydanticAIStructuredLLM`,
    :class:`~omniadapters.services.cove.testing.ScriptedLLM`, and the decorators in this module.
    """

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Return the model's answer to ``messages``, validated as ``response_model``."""
        ...


@dataclass(frozen=True, slots=True)
class LLMStep[BriefT, ResponseT: BaseModel, ResultT]:
    """One structured LLM call as a step: render the brief, call the model, parse the response.

    Build one per role with :func:`llm_step` and reuse it. Every call the step makes, including a
    witness fan-out, shares the role's single client.
    """

    llm: StructuredLLM
    prompt: Prompt[BriefT]
    response_model: type[ResponseT]
    parse: Callable[[ResponseT], ResultT]

    @property
    def template_variables(self) -> frozenset[str] | None:
        """The names the prompt's templates reference, or ``None`` if the prompt is an opaque function."""
        return self.prompt.variables if isinstance(self.prompt, JinjaPrompt) else None

    async def __call__(self, brief: BriefT) -> ResultT:
        """Run the call for ``brief``."""
        response = await self.llm.acreate(self.prompt(brief), self.response_model)
        return self.parse(response)


def _unchanged[ValueT](value: ValueT) -> ValueT:
    return value


@overload
def llm_step[BriefT, ResponseT: BaseModel](
    *, llm: StructuredLLM, prompt: Prompt[BriefT], response_model: type[ResponseT]
) -> LLMStep[BriefT, ResponseT, ResponseT]: ...


@overload
def llm_step[BriefT, ResponseT: BaseModel, ResultT](
    *,
    llm: StructuredLLM,
    prompt: Prompt[BriefT],
    response_model: type[ResponseT],
    parse: Callable[[ResponseT], ResultT],
) -> LLMStep[BriefT, ResponseT, ResultT]: ...


def llm_step[BriefT, ResponseT: BaseModel, ResultT](
    *,
    llm: StructuredLLM,
    prompt: Prompt[BriefT],
    response_model: type[ResponseT],
    parse: Callable[[ResponseT], ResultT] | None = None,
) -> LLMStep[BriefT, ResponseT, ResultT] | LLMStep[BriefT, ResponseT, ResponseT]:
    """Build an LLM-backed step.

    Parameters
    ----------
    llm
        The role's model. Pass the same object to every step the role plays.
    prompt
        Renders the brief into chat messages: a :class:`~omniadapters.services.cove.prompts.JinjaPrompt`
        or any ``(brief) -> Messages`` function.
    response_model
        The pydantic model the LLM must return.
    parse
        Maps the response to the step's result. Without it the step returns the response itself.

    Returns
    -------
    LLMStep
        A step, usable wherever the engine or a procedure expects one.

    """
    if parse is None:
        return LLMStep(llm=llm, prompt=prompt, response_model=response_model, parse=_unchanged)
    return LLMStep(llm=llm, prompt=prompt, response_model=response_model, parse=parse)


def check_prompt_fits(step: object, brief: type[DataclassInstance], *, role: Role) -> None:
    """Reject an LLM step whose template references a name the role's ``brief`` does not carry.

    Only :class:`LLMStep` objects with a :class:`~omniadapters.services.cove.prompts.JinjaPrompt` can be
    inspected; any other step is opaque and passes. Typed prompt functions get the same guarantee
    from the type checker instead.

    Raises
    ------
    TemplateVariableError
        If the templates reference a name the brief does not carry.

    """
    variables = step.template_variables if isinstance(step, LLMStep) else None
    if variables is None:
        return
    allowed = frozenset(item.name for item in dataclasses.fields(brief))
    unknown = variables - allowed
    if unknown:
        raise TemplateVariableError(role=role, brief=brief.__name__, unknown=unknown, allowed=allowed)


class AsyncLimiter(Protocol):
    """Anything usable as ``async with limiter:``: an ``asyncio.Semaphore``, a token bucket, a rate limiter."""

    async def __aenter__(self) -> object:
        """Wait for a slot."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
        /,
    ) -> bool | None:
        """Release the slot."""
        ...


@dataclass(frozen=True, slots=True)
class Throttled:
    """Cap concurrent calls to ``inner``: share one ``limiter`` across roles, runs and engines."""

    inner: StructuredLLM
    limiter: AsyncLimiter

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Call ``inner`` once a slot is free."""
        async with self.limiter:
            return await self.inner.acreate(messages, response_model)


class CallBudget:
    """A number of LLM calls, shared by every :class:`Budgeted` model that wraps it.

    Spending is synchronous, so a budget is safe to share across tasks on one event loop.
    """

    __slots__ = ("_max_calls", "_spent")

    def __init__(self, max_calls: int) -> None:
        if max_calls < 1:
            msg = f"a call budget needs at least one call, got {max_calls}"
            raise ValueError(msg)
        self._max_calls = max_calls
        self._spent = 0

    @property
    def max_calls(self) -> int:
        """The number of calls the budget allows."""
        return self._max_calls

    @property
    def spent(self) -> int:
        """The number of calls made so far."""
        return self._spent

    def spend(self) -> None:
        """Record one call.

        Raises
        ------
        BudgetExceededError
            If the budget is already spent. The call is then not made.

        """
        if self._spent >= self._max_calls:
            raise BudgetExceededError(max_calls=self._max_calls)
        self._spent += 1


@dataclass(frozen=True, slots=True)
class Budgeted:
    """Spend one unit of ``budget`` per call to ``inner``; refuse to call once it is gone."""

    inner: StructuredLLM
    budget: CallBudget

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Spend one call, then call ``inner``."""
        self.budget.spend()
        return await self.inner.acreate(messages, response_model)
