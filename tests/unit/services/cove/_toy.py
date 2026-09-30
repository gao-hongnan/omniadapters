"""A string-typed toy recipe whose steps are plain async functions, for engine and procedure tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omniadapters.services.cove import ChainOfVerification, Factored

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniadapters.services.cove import (
        CaseBrief,
        ContrarianBrief,
        JudgeBrief,
        Limits,
        Observer,
        Procedure,
        Retrial,
        Step,
        WitnessBrief,
    )

type ToyCove = ChainOfVerification[str, str, str, str, str]


async def propose(brief: CaseBrief[str]) -> str:
    return f"proposal for {brief.case}"


def contrarian_raising(*challenges: str) -> Step[ContrarianBrief[str, str], tuple[str, ...]]:
    async def challenge(brief: ContrarianBrief[str, str]) -> tuple[str, ...]:
        del brief
        return challenges

    return challenge


async def honest_witness(brief: WitnessBrief[str, str]) -> str:
    return f"answer to {brief.challenge}"


async def summarising_judge(brief: JudgeBrief[str, str, str, str]) -> str:
    answered = ", ".join(f"{item.challenge}={item.testimony}" for item in brief.evidence.answered)
    unanswered = ", ".join(item.challenge for item in brief.evidence.unanswered)
    return f"ruling on {brief.proposal} [{answered}] missing [{unanswered}]"


async def wait_forever() -> None:
    await asyncio.Event().wait()


def toy_cove(  # noqa: PLR0913 - mirrors the engine's keywords so each test overrides only what it needs
    *,
    proponent: Step[CaseBrief[str], str] = propose,
    procedure: Procedure[str, str, str, str] | None = None,
    judge: Step[JudgeBrief[str, str, str, str], str] = summarising_judge,
    retrial: Retrial[str, str] | None = None,
    limits: Limits | None = None,
    observers: Sequence[Observer] = (),
) -> ToyCove:
    examination = procedure or Factored(contrarian=contrarian_raising("a", "b", "c"), witness=honest_witness)
    if limits is None:
        return ChainOfVerification(
            proponent=proponent, procedure=examination, judge=judge, retrial=retrial, observers=observers
        )
    return ChainOfVerification(
        proponent=proponent, procedure=examination, judge=judge, retrial=retrial, limits=limits, observers=observers
    )


@dataclass
class InFlight:
    """Counts how many instrumented calls run at once."""

    current: int = 0
    peak: int = 0
    started: list[str] = field(default_factory=list)

    async def hold(self, label: str, seconds: float = 0.01) -> None:
        self.started.append(label)
        self.current += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.current -= 1
