"""The chain-of-verification engine: it owns the proceedings, recipes own the roles."""

from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .briefs import CaseBrief, JudgeBrief
from .court import Court, _Session
from .enums import Disposition, Role
from .errors import CoVeRunError
from .events import RunFailed, RunFinished, RunId, RunStarted
from .limits import DEFAULT_LIMITS
from .llm import check_prompt_fits
from .transcript import Hearing, Transcript

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .briefs import Step
    from .events import Observer
    from .limits import Limits
    from .procedures import Procedure


@dataclass(frozen=True, slots=True)
class Reopen[ProposalT]:
    """Ask for another hearing on a revised proposal."""

    proposal: ProposalT


@dataclass(frozen=True, slots=True)
class Retrial[ProposalT, RulingT]:
    """Opt-in multi-hearing chain of verification.

    After every ruling the engine asks ``reconsider``: returning :class:`Reopen` starts another
    hearing on the revised proposal, returning ``None`` lets the ruling stand. The engine never
    holds more than ``max_hearings`` hearings.
    """

    max_hearings: int
    reconsider: Callable[[RulingT], Reopen[ProposalT] | None]

    def __post_init__(self) -> None:
        if self.max_hearings < 2:  # noqa: PLR2004 - a retrial is at least a second hearing
            msg = f"a retrial needs max_hearings >= 2 (a single hearing is the default), got {self.max_hearings}"
            raise ValueError(msg)


class ChainOfVerification[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT]:
    """Chain-of-Verification (Dhuliawala et al. 2023, arXiv:2309.11495) as a reusable engine.

    The engine owns the proceedings: the proponent drafts a proposal, the procedure examines it,
    the judge rules, and an optional retrial repeats the hearing. It also owns everything
    around them: concurrency, deadlines, the quorum, tracing, and the round loop. A recipe
    supplies only the roles, as steps typed by the briefs they receive.

    Parameters
    ----------
    proponent
        Drafts the baseline response from the case.
    procedure
        How each hearing gathers evidence: :class:`~omniadapters.services.cove.procedures.Factored`,
        ``TwoStep``, ``Joint``, ``FactorRevise``, or your own
        :class:`~omniadapters.services.cove.procedures.Procedure`.
    judge
        Rules on the proposal given the evidence.
    retrial
        Optional: re-examine revised proposals.
    limits
        Every bound the engine enforces.
    observers
        Callables that receive every event. They must be fast; an observer that raises is logged
        and ignored.

    """

    def __init__(  # noqa: PLR0913 - one keyword per courtroom role plus three opt-in policies
        self,
        *,
        proponent: Step[CaseBrief[CaseT], ProposalT],
        procedure: Procedure[CaseT, ProposalT, ChallengeT, TestimonyT],
        judge: Step[JudgeBrief[CaseT, ProposalT, ChallengeT, TestimonyT], RulingT],
        retrial: Retrial[ProposalT, RulingT] | None = None,
        limits: Limits = DEFAULT_LIMITS,
        observers: Sequence[Observer] = (),
    ) -> None:
        check_prompt_fits(proponent, CaseBrief, role=Role.PROPONENT)
        check_prompt_fits(judge, JudgeBrief, role=Role.JUDGE)
        self._proponent = proponent
        self._procedure = procedure
        self._judge = judge
        self._retrial = retrial
        self._limits = limits
        self._observers = tuple(observers)
        # One step semaphore per event loop, so an engine can be reused across `asyncio.run` calls.
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def limits(self) -> Limits:
        """The bounds this engine enforces."""
        return self._limits

    async def run(self, case: CaseT) -> Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT]:
        """Try ``case`` and return the transcript.

        Raises
        ------
        CoVeRunError
            If a step fails, the contrarian's challenges are out of bounds, a panel miscounts, or
            the quorum is lost. ``error.record`` holds the steps that ran first.

        """
        session = _Session(
            run_id=RunId(uuid.uuid4().hex),
            limits=self._limits,
            semaphore=self._step_semaphore(),
            observers=self._observers,
        )
        session.emit(RunStarted(run_id=session.run_id))
        started = time.perf_counter()
        try:
            transcript = await self._proceed(session, case)
        except Exception as exc:
            if isinstance(exc, CoVeRunError):
                exc.record = tuple(session.records)
            session.emit(
                RunFailed(
                    run_id=session.run_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    elapsed_s=time.perf_counter() - started,
                )
            )
            raise
        session.emit(
            RunFinished(
                run_id=session.run_id,
                disposition=transcript.disposition,
                hearings=len(transcript.hearings),
                elapsed_s=time.perf_counter() - started,
            )
        )
        return transcript

    async def run_many(
        self, cases: Sequence[CaseT]
    ) -> list[Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT] | CoVeRunError]:
        """Try a batch of cases, ``limits.max_concurrent_runs`` at a time, preserving their order.

        A case whose run fails comes back as its :class:`CoVeRunError`, in place; the batch
        carries on. Any other exception is a bug and aborts the batch.
        """
        results: dict[int, Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT] | CoVeRunError] = {}
        pending = iter(enumerate(cases))

        async def work_through_docket() -> None:
            for index, case in pending:
                results[index] = await self._run_or_error(case)

        async with asyncio.TaskGroup() as group:
            for _ in range(min(self._limits.max_concurrent_runs, len(cases))):
                group.create_task(work_through_docket())
        return [results[index] for index in range(len(cases))]

    async def _run_or_error(
        self, case: CaseT
    ) -> Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT] | CoVeRunError:
        try:
            return await self.run(case)
        except CoVeRunError as exc:
            return exc

    async def _proceed(
        self, session: _Session, case: CaseT
    ) -> Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT]:
        proposal = await Court(session, hearing=1).call(self._proponent, CaseBrief(case=case), role=Role.PROPONENT)
        max_hearings = 1 if self._retrial is None else self._retrial.max_hearings
        hearings: list[Hearing[ProposalT, ChallengeT, TestimonyT, RulingT]] = []
        disposition = Disposition.HEARING_LIMIT
        for number in range(1, max_hearings + 1):
            court = Court(session, hearing=number)
            evidence = await self._procedure.examine(court, case=case, proposal=proposal)
            ruling = await court.call(
                self._judge,
                JudgeBrief(case=case, proposal=proposal, evidence=evidence, hearing=number),
                role=Role.JUDGE,
            )
            hearings.append(Hearing(number=number, proposal=proposal, evidence=evidence, ruling=ruling))
            if self._retrial is None:
                disposition = Disposition.SINGLE_HEARING
                break
            reopened = self._retrial.reconsider(ruling)
            if reopened is None:
                disposition = Disposition.SETTLED
                break
            proposal = reopened.proposal
        return Transcript(
            run_id=session.run_id,
            case=case,
            hearings=tuple(hearings),
            disposition=disposition,
            record=tuple(session.records),
        )

    def _step_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._limits.max_concurrent_steps)
            self._semaphores[loop] = semaphore
        return semaphore
