"""Errors raised by :mod:`omniadapters.services.cove`.

- :class:`CoVeError` is the base of everything below; ``context`` carries structured fields for logs.
- :class:`TemplateVariableError` is raised while a recipe is *built*, before any call is made.
- :class:`CoVeRunError` and its subclasses are raised when a *run* fails. ``ChainOfVerification.run_many``
  returns them in place, so one failed case never sinks a batch.
- :class:`StepTimeoutError` and :class:`BudgetExceededError` are raised *inside* a step. The engine turns them
  into an ``Unanswered`` outcome, for a witness, or into :class:`StepFailedError`, for any other role.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .enums import Role
    from .events import RunId, StepRecord
    from .evidence import Unanswered


class CoVeError(Exception):
    """Base of every error this package raises."""

    message: str
    context: Mapping[str, object]

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = MappingProxyType(context)


class TemplateVariableError(CoVeError, ValueError):
    """A prompt template references a name its role's brief does not carry.

    Raised when a recipe is built, before any LLM call, so a witness prompt that mentions
    ``proposal`` (or a misspelt field) fails immediately.
    """

    role: Role
    unknown: frozenset[str]
    allowed: frozenset[str]

    def __init__(self, *, role: Role, brief: str, unknown: frozenset[str], allowed: frozenset[str]) -> None:
        message = f"{role} prompt references {sorted(unknown)}, but a {brief} carries only {sorted(allowed)}"
        super().__init__(message, role=role, brief=brief, unknown=sorted(unknown), allowed=sorted(allowed))
        self.role = role
        self.unknown = unknown
        self.allowed = allowed


class BudgetExceededError(CoVeError):
    """A shared :class:`~omniadapters.services.cove.llm.CallBudget` is spent; the LLM call was not made."""

    max_calls: int

    def __init__(self, *, max_calls: int) -> None:
        super().__init__(f"the call budget of {max_calls} LLM call(s) is spent", max_calls=max_calls)
        self.max_calls = max_calls


class StepTimeoutError(CoVeError, TimeoutError):
    """A step ran past its deadline and was cancelled."""

    role: Role
    timeout_s: float

    def __init__(self, *, role: Role, timeout_s: float) -> None:
        super().__init__(f"{role} step exceeded its {timeout_s:g}s timeout", role=role, timeout_s=timeout_s)
        self.role = role
        self.timeout_s = timeout_s


class CoVeRunError(CoVeError):
    """A run failed.

    Attributes
    ----------
    run_id
        The failed run.
    hearing
        The hearing it failed in, counted from 1.
    record
        The steps the run completed or failed before the error. The engine fills it in before
        the error leaves ``ChainOfVerification.run``.

    """

    run_id: RunId
    hearing: int
    record: tuple[StepRecord, ...]

    def __init__(self, message: str, *, run_id: RunId, hearing: int, **context: object) -> None:
        super().__init__(f"[run {run_id} hearing {hearing}] {message}", run_id=run_id, hearing=hearing, **context)
        self.run_id = run_id
        self.hearing = hearing
        self.record = ()


class StepFailedError(CoVeRunError):
    """A proponent, contrarian, panel, inquisitor or judge step failed; the cause is chained."""

    role: Role

    def __init__(self, *, run_id: RunId, hearing: int, role: Role, cause: BaseException) -> None:
        super().__init__(
            f"the {role} step failed: {type(cause).__name__}: {cause}",
            run_id=run_id,
            hearing=hearing,
            role=role,
            cause=type(cause).__name__,
        )
        self.role = role


class ChallengeCountError(CoVeRunError):
    """The contrarian raised fewer or more challenges than the limits allow."""

    count: int
    minimum: int
    maximum: int

    def __init__(self, *, run_id: RunId, hearing: int, count: int, minimum: int, maximum: int) -> None:
        super().__init__(
            f"the contrarian raised {count} challenge(s); the limits allow {minimum} to {maximum}",
            run_id=run_id,
            hearing=hearing,
            count=count,
            minimum=minimum,
            maximum=maximum,
        )
        self.count = count
        self.minimum = minimum
        self.maximum = maximum


class PanelMismatchError(CoVeRunError):
    """A two-step panel returned a different number of testimonies than there were challenges."""

    challenges: int
    testimonies: int

    def __init__(self, *, run_id: RunId, hearing: int, challenges: int, testimonies: int) -> None:
        super().__init__(
            f"the panel returned {testimonies} testimonies for {challenges} challenge(s)",
            run_id=run_id,
            hearing=hearing,
            challenges=challenges,
            testimonies=testimonies,
        )
        self.challenges = challenges
        self.testimonies = testimonies


class QuorumNotMetError(CoVeRunError):
    """More challenges went unanswered than the quorum tolerates; the remaining witnesses were dismissed."""

    unanswered: tuple[Unanswered[object], ...]
    total: int
    tolerated: int

    def __init__(
        self,
        *,
        run_id: RunId,
        hearing: int,
        unanswered: tuple[Unanswered[object], ...],
        total: int,
        tolerated: int,
    ) -> None:
        details = "; ".join(f"{item.challenge!r} -> {item.error_type}: {item.message}" for item in unanswered)
        super().__init__(
            f"{len(unanswered)} of {total} challenge(s) went unanswered; the quorum tolerates {tolerated}: {details}",
            run_id=run_id,
            hearing=hearing,
            unanswered=len(unanswered),
            total=total,
            tolerated=tolerated,
        )
        self.unanswered = unanswered
        self.total = total
        self.tolerated = tolerated
