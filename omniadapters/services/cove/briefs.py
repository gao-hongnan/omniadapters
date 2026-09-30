"""Briefs: what each role is allowed to see.

A step is ``async (brief) -> result``. The brief's type is the information-hiding contract of
Chain-of-Verification: the paper's central finding is that verification answers conditioned on
the draft repeat its hallucinations, so :class:`WitnessBrief` and :class:`PanelBrief` have no
``proposal`` field. A witness cannot be conditioned on the proposal because the value is never
handed to it; the type checker rejects ``brief.proposal`` in a witness step, and the engine
rejects a witness prompt that references ``proposal`` when the recipe is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .evidence import Evidence

type Step[BriefT, ResultT] = Callable[[BriefT], Awaitable[ResultT]]


@dataclass(frozen=True, slots=True)
class CaseBrief[CaseT]:
    """What the proponent sees, and a blind contrarian: the case alone."""

    case: CaseT


@dataclass(frozen=True, slots=True)
class ContrarianBrief[CaseT, ProposalT]:
    """What the contrarian and the joint-variant inquisitor see: the case and the proposal under challenge."""

    case: CaseT
    proposal: ProposalT


@dataclass(frozen=True, slots=True)
class WitnessBrief[CaseT, ChallengeT]:
    """What one sequestered witness sees: the case and a single challenge, never the proposal."""

    case: CaseT
    challenge: ChallengeT


@dataclass(frozen=True, slots=True)
class PanelBrief[CaseT, ChallengeT]:
    """What the two-step panel sees: the case and every challenge, never the proposal."""

    case: CaseT
    challenges: tuple[ChallengeT, ...]


@dataclass(frozen=True, slots=True)
class CrossExaminationBrief[CaseT, ProposalT, ChallengeT, TestimonyT]:
    """What the cross-examiner sees: one independent testimony next to the proposal it may contradict."""

    case: CaseT
    proposal: ProposalT
    challenge: ChallengeT
    testimony: TestimonyT


@dataclass(frozen=True, slots=True)
class JudgeBrief[CaseT, ProposalT, ChallengeT, TestimonyT]:
    """What the judge sees: the case, the proposal, the evidence, and which hearing this is (from 1)."""

    case: CaseT
    proposal: ProposalT
    evidence: Evidence[ChallengeT, TestimonyT]
    hearing: int
