"""What a chain-of-verification run reports while it runs, and what its transcript records.

Every event is a frozen dataclass with a ``kind`` discriminator, so the stream is easy to
render in a UI and the transcript's ``record`` serialises to self-describing JSON.
Observers are plain callables; the engine isolates them, so an observer that raises is
logged and never fails the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, NewType, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable

    from .enums import Disposition, Role

RunId = NewType("RunId", str)

_LOGGER: Final = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunStarted:
    """A run began."""

    run_id: RunId
    kind: Literal["run_started"] = field(default="run_started", init=False)


@dataclass(frozen=True, slots=True)
class StepStarted:
    """A role's step was called. ``challenge_index`` is set for witness and cross-examiner calls."""

    run_id: RunId
    role: Role
    hearing: int
    challenge_index: int | None
    kind: Literal["step_started"] = field(default="step_started", init=False)


@dataclass(frozen=True, slots=True)
class StepCompleted:
    """A role's step returned."""

    run_id: RunId
    role: Role
    hearing: int
    challenge_index: int | None
    elapsed_s: float
    kind: Literal["step_completed"] = field(default="step_completed", init=False)


@dataclass(frozen=True, slots=True)
class StepRaised:
    """A role's step raised, timed out, or was cancelled."""

    run_id: RunId
    role: Role
    hearing: int
    challenge_index: int | None
    elapsed_s: float
    error_type: str
    message: str
    kind: Literal["step_raised"] = field(default="step_raised", init=False)


@dataclass(frozen=True, slots=True)
class RunFinished:
    """A run produced its final ruling."""

    run_id: RunId
    disposition: Disposition
    hearings: int
    elapsed_s: float
    kind: Literal["run_finished"] = field(default="run_finished", init=False)


@dataclass(frozen=True, slots=True)
class RunFailed:
    """A run raised instead of ruling."""

    run_id: RunId
    error_type: str
    message: str
    elapsed_s: float
    kind: Literal["run_failed"] = field(default="run_failed", init=False)


type StepRecord = StepCompleted | StepRaised
type CoVeEvent = RunStarted | StepStarted | StepCompleted | StepRaised | RunFinished | RunFailed
type Observer = Callable[[CoVeEvent], None]


def logging_observer(event: CoVeEvent) -> None:
    """Log ``event`` to this module's logger: failures at WARNING, everything else at DEBUG."""
    match event:
        case RunStarted():
            _LOGGER.debug("cove run %s started", event.run_id)
        case StepStarted():
            _LOGGER.debug(
                "cove run %s hearing %d: %s%s started",
                event.run_id,
                event.hearing,
                event.role,
                _position(event.challenge_index),
            )
        case StepCompleted():
            _LOGGER.debug(
                "cove run %s hearing %d: %s%s completed in %.3fs",
                event.run_id,
                event.hearing,
                event.role,
                _position(event.challenge_index),
                event.elapsed_s,
            )
        case StepRaised():
            _LOGGER.warning(
                "cove run %s hearing %d: %s%s raised %s after %.3fs: %s",
                event.run_id,
                event.hearing,
                event.role,
                _position(event.challenge_index),
                event.error_type,
                event.elapsed_s,
                event.message,
            )
        case RunFinished():
            _LOGGER.debug(
                "cove run %s finished (%s) after %d hearing(s) in %.3fs",
                event.run_id,
                event.disposition,
                event.hearings,
                event.elapsed_s,
            )
        case RunFailed():
            _LOGGER.warning(
                "cove run %s failed with %s after %.3fs: %s",
                event.run_id,
                event.error_type,
                event.elapsed_s,
                event.message,
            )
        case _:
            assert_never(event)


def _position(challenge_index: int | None) -> str:
    return "" if challenge_index is None else f"[{challenge_index}]"
