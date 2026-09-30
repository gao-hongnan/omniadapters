"""The court: the runtime a procedure examines a proposal through.

A procedure never calls a step directly. It goes through its hearing's :class:`Court` (for
whole-hearing steps) or a :class:`Stand` (for one challenge inside :meth:`Court.hear_each`), so
every call is bounded, recorded and traced the same way. That is what lets customers write their
own procedures: they inherit the engine's concurrency cap, deadlines, quorum and audit trail by
calling through the court.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Final

from .enums import UnansweredReason
from .errors import ChallengeCountError, QuorumNotMetError, StepFailedError, StepTimeoutError
from .events import StepCompleted, StepRaised, StepStarted
from .evidence import Answered, Evidence, Unanswered

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from .briefs import Step
    from .enums import Role
    from .events import CoVeEvent, Observer, RunId, StepRecord
    from .evidence import Outcome
    from .limits import Limits

_LOGGER: Final = logging.getLogger(__name__)


class _QuorumLostError(Exception):
    """Raised inside a hearing task so the task group dismisses the remaining witnesses."""


class _Session:
    """Private per-run state shared by every court and stand of one run."""

    __slots__ = ("limits", "observers", "records", "run_id", "semaphore")

    def __init__(
        self,
        *,
        run_id: RunId,
        limits: Limits,
        semaphore: asyncio.Semaphore,
        observers: tuple[Observer, ...],
    ) -> None:
        self.run_id = run_id
        self.limits = limits
        self.semaphore = semaphore
        self.observers = observers
        self.records: list[StepRecord] = []

    def emit(self, event: CoVeEvent) -> None:
        for observer in self.observers:
            try:
                observer(event)
            except Exception:
                _LOGGER.exception("cove observer %r failed on %r; the run continues", observer, event)

    async def invoke[BriefT, ResultT](  # noqa: PLR0913 - one keyword per field of the step's events
        self,
        step: Step[BriefT, ResultT],
        brief: BriefT,
        *,
        role: Role,
        hearing: int,
        challenge_index: int | None,
        timeout_s: float,
    ) -> ResultT:
        """Call ``step`` under the engine's concurrency cap and a deadline, recording the outcome."""
        async with self.semaphore:
            self.emit(StepStarted(run_id=self.run_id, role=role, hearing=hearing, challenge_index=challenge_index))
            started = time.perf_counter()
            deadline = asyncio.timeout(timeout_s)
            try:
                async with deadline:
                    result = await step(brief)
            except asyncio.CancelledError:
                self._raised(role, hearing, challenge_index, started, "CancelledError", "cancelled")
                raise
            except Exception as exc:
                if isinstance(exc, TimeoutError) and deadline.expired():
                    timed_out = StepTimeoutError(role=role, timeout_s=timeout_s)
                    self._raised(role, hearing, challenge_index, started, type(timed_out).__name__, str(timed_out))
                    raise timed_out from exc
                self._raised(role, hearing, challenge_index, started, type(exc).__name__, str(exc))
                raise
        self._record(
            StepCompleted(
                run_id=self.run_id,
                role=role,
                hearing=hearing,
                challenge_index=challenge_index,
                elapsed_s=time.perf_counter() - started,
            )
        )
        return result

    def _raised(  # noqa: PLR0913 - the fields of one StepRaised event
        self,
        role: Role,
        hearing: int,
        challenge_index: int | None,
        started: float,
        error_type: str,
        message: str,
    ) -> None:
        self._record(
            StepRaised(
                run_id=self.run_id,
                role=role,
                hearing=hearing,
                challenge_index=challenge_index,
                elapsed_s=time.perf_counter() - started,
                error_type=error_type,
                message=message,
            )
        )

    def _record(self, record: StepRecord) -> None:
        self.records.append(record)
        self.emit(record)


class Court:
    """One hearing's handle for calling steps, admitting challenges and hearing witnesses.

    The engine creates a court per hearing and hands it to the procedure's ``examine``.
    """

    __slots__ = ("_hearing", "_session")

    def __init__(self, session: _Session, *, hearing: int) -> None:
        self._session = session
        self._hearing = hearing

    @property
    def run_id(self) -> RunId:
        """The run this hearing belongs to."""
        return self._session.run_id

    @property
    def hearing(self) -> int:
        """The hearing's number, counted from 1."""
        return self._hearing

    async def call[BriefT, ResultT](self, step: Step[BriefT, ResultT], brief: BriefT, *, role: Role) -> ResultT:
        """Call a whole-hearing step (proponent, contrarian, panel, inquisitor, judge) under ``limits.timeout_s``.

        Raises
        ------
        StepFailedError
            If the step raises or times out. The original error is chained as its cause.

        """
        try:
            return await self._session.invoke(
                step,
                brief,
                role=role,
                hearing=self._hearing,
                challenge_index=None,
                timeout_s=self._session.limits.timeout_s,
            )
        except Exception as exc:
            raise StepFailedError(run_id=self.run_id, hearing=self._hearing, role=role, cause=exc) from exc

    def admit[ItemT](self, challenges: Sequence[ItemT]) -> tuple[ItemT, ...]:
        """Admit the contrarian's challenges if their number is within the limits.

        Raises
        ------
        ChallengeCountError
            If there are fewer than ``limits.min_challenges`` or more than ``limits.max_challenges``.

        """
        admitted = tuple(challenges)
        limits = self._session.limits
        if not limits.min_challenges <= len(admitted) <= limits.max_challenges:
            raise ChallengeCountError(
                run_id=self.run_id,
                hearing=self._hearing,
                count=len(admitted),
                minimum=limits.min_challenges,
                maximum=limits.max_challenges,
            )
        return admitted

    async def hear_each[ChallengeT, TestimonyT](
        self,
        challenges: Sequence[ChallengeT],
        examine: Callable[[Stand, ChallengeT], Awaitable[TestimonyT]],
    ) -> Evidence[ChallengeT, TestimonyT]:
        """Examine every challenge concurrently, each on its own :class:`Stand`, and enforce the quorum.

        A challenge whose examination raises or times out becomes an ``Unanswered`` outcome.
        Cancellation is never converted: it propagates.

        Raises
        ------
        QuorumNotMetError
            As soon as more challenges go unanswered than ``limits.quorum`` tolerates. The remaining
            examinations are cancelled.

        """
        tolerated = self._session.limits.tolerated_unanswered(len(challenges))
        outcomes: dict[int, Outcome[ChallengeT, TestimonyT]] = {}
        missing: dict[int, Unanswered[ChallengeT]] = {}

        async def hear(index: int, challenge: ChallengeT) -> None:
            stand = Stand(self._session, hearing=self._hearing, challenge_index=index)
            try:
                testimony = await examine(stand, challenge)
            except Exception as exc:  # becomes a typed Unanswered outcome; the quorum decides
                reason = UnansweredReason.TIMEOUT if isinstance(exc, TimeoutError) else UnansweredReason.ERROR
                unanswered = Unanswered(
                    challenge=challenge, reason=reason, error_type=type(exc).__name__, message=str(exc)
                )
                outcomes[index] = unanswered
                missing[index] = unanswered
                if len(missing) > tolerated:
                    raise _QuorumLostError from exc
            else:
                outcomes[index] = Answered(challenge=challenge, testimony=testimony)

        quorum_lost = False
        try:
            async with asyncio.TaskGroup() as group:
                for index, challenge in enumerate(challenges):
                    group.create_task(hear(index, challenge))
        except* _QuorumLostError:
            quorum_lost = True
        if quorum_lost:
            raise QuorumNotMetError(
                run_id=self.run_id,
                hearing=self._hearing,
                unanswered=tuple(_widen(missing[index]) for index in sorted(missing)),
                total=len(challenges),
                tolerated=tolerated,
            )
        return Evidence(outcomes=tuple(outcomes[index] for index in range(len(challenges))))


class Stand:
    """One challenge's handle inside :meth:`Court.hear_each`.

    Steps called through a stand run under ``limits.witness_timeout_s``. A failure propagates to
    ``hear_each``, which records the challenge as unanswered instead of ending the hearing.
    """

    __slots__ = ("_challenge_index", "_hearing", "_session")

    def __init__(self, session: _Session, *, hearing: int, challenge_index: int) -> None:
        self._session = session
        self._hearing = hearing
        self._challenge_index = challenge_index

    @property
    def challenge_index(self) -> int:
        """The position of this stand's challenge in the hearing, from 0."""
        return self._challenge_index

    async def call[BriefT, ResultT](self, step: Step[BriefT, ResultT], brief: BriefT, *, role: Role) -> ResultT:
        """Call a per-challenge step (witness, cross-examiner) under ``limits.witness_timeout_s``."""
        return await self._session.invoke(
            step,
            brief,
            role=role,
            hearing=self._hearing,
            challenge_index=self._challenge_index,
            timeout_s=self._session.limits.witness_timeout_s,
        )


def _widen[ChallengeT](unanswered: Unanswered[ChallengeT]) -> Unanswered[object]:
    # QuorumNotMetError outlives the procedure's type parameters, and frozen dataclasses are invariant.
    return Unanswered(
        challenge=unanswered.challenge,
        reason=unanswered.reason,
        error_type=unanswered.error_type,
        message=unanswered.message,
    )
