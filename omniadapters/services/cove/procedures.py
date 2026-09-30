"""Procedures: how a hearing examines a proposal. The four execution variants of the CoVe paper.

======================  ======================================  ===============================
Procedure               Calls per hearing                       Who sees the proposal
======================  ======================================  ===============================
:class:`Factored`       contrarian + one witness per challenge  contrarian
:class:`TwoStep`        contrarian + one panel                  contrarian
:class:`Joint`          one inquisitor                          inquisitor (it may copy errors)
:class:`FactorRevise`   factored + one cross-examiner each      contrarian, cross-examiners
======================  ======================================  ===============================

No variant wins everywhere (Dhuliawala et al. 2023, section 4), so the procedure is a strategy
the recipe picks. It is also an open set: implement :class:`Procedure` and call steps through
the :class:`~omniadapters.services.cove.court.Court` to get the engine's limits and tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .briefs import CaseBrief, ContrarianBrief, CrossExaminationBrief, PanelBrief, WitnessBrief
from .enums import Role
from .errors import PanelMismatchError
from .evidence import Answered, CrossExamined, Evidence
from .llm import check_prompt_fits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .briefs import Step
    from .court import Court, Stand


class Procedure[CaseT, ProposalT, ChallengeT, TestimonyT](Protocol):
    """How a hearing turns a proposal into evidence for the judge."""

    async def examine(self, court: Court, *, case: CaseT, proposal: ProposalT) -> Evidence[ChallengeT, TestimonyT]:
        """Examine ``proposal`` and return one outcome per challenge."""
        ...


@dataclass(frozen=True, slots=True)
class Factored[CaseT, ProposalT, ChallengeT, TestimonyT]:
    """Each challenge goes to its own sequestered witness, who sees the case and that challenge only.

    The paper's strongest plain variant: testimony cannot copy the proposal's hallucinations,
    because the proposal is never in a witness's brief.
    """

    contrarian: Step[ContrarianBrief[CaseT, ProposalT], Sequence[ChallengeT]]
    witness: Step[WitnessBrief[CaseT, ChallengeT], TestimonyT]

    def __post_init__(self) -> None:
        check_prompt_fits(self.contrarian, ContrarianBrief, role=Role.CONTRARIAN)
        check_prompt_fits(self.witness, WitnessBrief, role=Role.WITNESS)

    async def examine(self, court: Court, *, case: CaseT, proposal: ProposalT) -> Evidence[ChallengeT, TestimonyT]:
        """Challenge the proposal, then hear every challenge concurrently."""
        challenges = court.admit(
            await court.call(self.contrarian, ContrarianBrief(case=case, proposal=proposal), role=Role.CONTRARIAN)
        )

        async def testify(stand: Stand, challenge: ChallengeT) -> TestimonyT:
            return await stand.call(self.witness, WitnessBrief(case=case, challenge=challenge), role=Role.WITNESS)

        return await court.hear_each(challenges, testify)


@dataclass(frozen=True, slots=True)
class TwoStep[CaseT, ProposalT, ChallengeT, TestimonyT]:
    """The contrarian challenges once; a panel answers every challenge in one call, without the proposal."""

    contrarian: Step[ContrarianBrief[CaseT, ProposalT], Sequence[ChallengeT]]
    panel: Step[PanelBrief[CaseT, ChallengeT], Sequence[TestimonyT]]

    def __post_init__(self) -> None:
        check_prompt_fits(self.contrarian, ContrarianBrief, role=Role.CONTRARIAN)
        check_prompt_fits(self.panel, PanelBrief, role=Role.PANEL)

    async def examine(self, court: Court, *, case: CaseT, proposal: ProposalT) -> Evidence[ChallengeT, TestimonyT]:
        """Challenge the proposal, then put every challenge to the panel at once.

        Raises
        ------
        PanelMismatchError
            If the panel answers a different number of challenges than it was asked.

        """
        challenges = court.admit(
            await court.call(self.contrarian, ContrarianBrief(case=case, proposal=proposal), role=Role.CONTRARIAN)
        )
        testimonies = tuple(await court.call(self.panel, PanelBrief(case=case, challenges=challenges), role=Role.PANEL))
        if len(testimonies) != len(challenges):
            raise PanelMismatchError(
                run_id=court.run_id,
                hearing=court.hearing,
                challenges=len(challenges),
                testimonies=len(testimonies),
            )
        return Evidence(
            outcomes=tuple(
                Answered(challenge=challenge, testimony=testimony)
                for challenge, testimony in zip(challenges, testimonies, strict=True)
            )
        )


@dataclass(frozen=True, slots=True)
class Joint[CaseT, ProposalT, ChallengeT, TestimonyT]:
    """One inquisitor challenges and answers in a single call that sees the proposal.

    The cheapest variant, and the one most prone to repeating the proposal's errors: its answers
    are conditioned on the very text they check.
    """

    inquisitor: Step[ContrarianBrief[CaseT, ProposalT], Sequence[Answered[ChallengeT, TestimonyT]]]

    def __post_init__(self) -> None:
        check_prompt_fits(self.inquisitor, ContrarianBrief, role=Role.INQUISITOR)

    async def examine(self, court: Court, *, case: CaseT, proposal: ProposalT) -> Evidence[ChallengeT, TestimonyT]:
        """Challenge and answer in one call."""
        answered = court.admit(
            await court.call(self.inquisitor, ContrarianBrief(case=case, proposal=proposal), role=Role.INQUISITOR)
        )
        return Evidence(outcomes=answered)


@dataclass(frozen=True, slots=True)
class FactorRevise[CaseT, ProposalT, ChallengeT, TestimonyT, FindingT]:
    """Factored testimony, then a cross-examiner checks each testimony against the proposal.

    The paper's best variant on long-form answers: the testimony stays independent, and the
    explicit cross-examination tells the judge exactly where the proposal is contradicted.
    """

    contrarian: Step[ContrarianBrief[CaseT, ProposalT], Sequence[ChallengeT]]
    witness: Step[WitnessBrief[CaseT, ChallengeT], TestimonyT]
    cross_examiner: Step[CrossExaminationBrief[CaseT, ProposalT, ChallengeT, TestimonyT], FindingT]

    def __post_init__(self) -> None:
        check_prompt_fits(self.contrarian, ContrarianBrief, role=Role.CONTRARIAN)
        check_prompt_fits(self.witness, WitnessBrief, role=Role.WITNESS)
        check_prompt_fits(self.cross_examiner, CrossExaminationBrief, role=Role.CROSS_EXAMINER)

    async def examine(
        self, court: Court, *, case: CaseT, proposal: ProposalT
    ) -> Evidence[ChallengeT, CrossExamined[TestimonyT, FindingT]]:
        """Challenge the proposal, then hear and cross-examine every challenge concurrently."""
        challenges = court.admit(
            await court.call(self.contrarian, ContrarianBrief(case=case, proposal=proposal), role=Role.CONTRARIAN)
        )

        async def testify_then_cross_examine(
            stand: Stand, challenge: ChallengeT
        ) -> CrossExamined[TestimonyT, FindingT]:
            testimony = await stand.call(self.witness, WitnessBrief(case=case, challenge=challenge), role=Role.WITNESS)
            finding = await stand.call(
                self.cross_examiner,
                CrossExaminationBrief(case=case, proposal=proposal, challenge=challenge, testimony=testimony),
                role=Role.CROSS_EXAMINER,
            )
            return CrossExamined(statement=testimony, finding=finding)

        return await court.hear_each(challenges, testify_then_cross_examine)


def blind[CaseT, ProposalT, ResultT](
    contrarian: Step[CaseBrief[CaseT], ResultT],
) -> Step[ContrarianBrief[CaseT, ProposalT], ResultT]:
    """Let a contrarian challenge the case without seeing the proposal.

    For recipes whose challenges must not be shaped by the proposal: tests written from a
    specification rather than from the code under test, or rubric items drawn from a policy
    rather than from the assessment being checked.
    """
    check_prompt_fits(contrarian, CaseBrief, role=Role.CONTRARIAN)

    async def challenge_blind(brief: ContrarianBrief[CaseT, ProposalT]) -> ResultT:
        return await contrarian(CaseBrief(case=brief.case))

    return challenge_blind
