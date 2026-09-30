"""Procedures: the paper's four variants, what each role may see, the quorum, bounds, and custom procedures."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from omniadapters.services.cove import (
    Answered,
    CaseBrief,
    ChainOfVerification,
    ChallengeCountError,
    ContrarianBrief,
    CoVeEvent,
    CrossExaminationBrief,
    CrossExamined,
    Evidence,
    Factored,
    FactorRevise,
    JinjaPrompt,
    Joint,
    JudgeBrief,
    Limits,
    PanelBrief,
    PanelMismatchError,
    QuorumNotMetError,
    Role,
    TemplateVariableError,
    TwoStep,
    Unanswered,
    UnansweredReason,
    WitnessBrief,
    blind,
    llm_step,
)
from omniadapters.services.cove.events import StepRaised
from omniadapters.services.cove.testing import ScriptedLLM
from tests.unit.services.cove._toy import contrarian_raising, honest_witness, propose, toy_cove, wait_forever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniadapters.services.cove import Court, Procedure, Stand, Step
    from omniadapters.services.cove.testing import RecordedCall


class _Reply(BaseModel):
    text: str


def _never_called(call: RecordedCall) -> BaseModel:
    raise AssertionError(call)


def _llm_prompted(user: str) -> Step[object, list[str]]:
    """Build an LLM step that is only ever checked when a recipe is built, never called."""
    return llm_step(
        llm=ScriptedLLM(_never_called),
        prompt=JinjaPrompt(system="role", user=user),
        response_model=_Reply,
        parse=lambda reply: [reply.text],
    )


@pytest.mark.unit
class TestIsolation:
    def test_witness_and_panel_briefs_carry_no_proposal(self) -> None:
        assert {item.name for item in dataclasses.fields(WitnessBrief)} == {"case", "challenge"}
        assert {item.name for item in dataclasses.fields(PanelBrief)} == {"case", "challenges"}

    @pytest.mark.asyncio
    async def test_witnesses_are_never_handed_the_proposal(self) -> None:
        briefs: list[WitnessBrief[str, str]] = []

        async def recording_witness(brief: WitnessBrief[str, str]) -> str:
            briefs.append(brief)
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b"), witness=recording_witness)
        await toy_cove(procedure=procedure).run("q")

        assert briefs
        assert all("proposal for q" not in repr(brief) for brief in briefs)

    def test_a_witness_prompt_naming_the_proposal_is_rejected_when_built(self) -> None:
        with pytest.raises(TemplateVariableError) as caught:
            Factored(contrarian=contrarian_raising("a"), witness=_llm_prompted("{{ challenge }} vs {{ proposal }}"))

        assert caught.value.role is Role.WITNESS
        assert caught.value.unknown == frozenset({"proposal"})
        assert caught.value.allowed == frozenset({"case", "challenge"})

    def test_a_panel_prompt_naming_the_proposal_is_rejected_when_built(self) -> None:
        with pytest.raises(TemplateVariableError) as caught:
            TwoStep(contrarian=contrarian_raising("a"), panel=_llm_prompted("{{ challenges }} {{ proposal }}"))

        assert caught.value.role is Role.PANEL

    def test_a_misspelt_contrarian_variable_is_rejected_when_built(self) -> None:
        with pytest.raises(TemplateVariableError, match="propsal"):
            Factored(contrarian=_llm_prompted("{{ propsal }}"), witness=honest_witness)

    def test_a_cross_examiner_may_see_the_proposal(self) -> None:
        FactorRevise(
            contrarian=contrarian_raising("a"),
            witness=honest_witness,
            cross_examiner=_llm_prompted("{{ proposal }} {{ challenge }} {{ testimony }} {{ case }}"),
        )

    @pytest.mark.asyncio
    async def test_a_blind_contrarian_sees_only_the_case(self) -> None:
        seen: list[CaseBrief[str]] = []

        async def write_tests_from_the_spec(brief: CaseBrief[str]) -> tuple[str, ...]:
            seen.append(brief)
            return ("t1", "t2")

        procedure: Procedure[str, str, str, str] = Factored(
            contrarian=blind(write_tests_from_the_spec), witness=honest_witness
        )
        transcript = await toy_cove(procedure=procedure).run("spec")

        assert seen == [CaseBrief(case="spec")]
        assert [outcome.challenge for outcome in transcript.hearings[0].evidence.outcomes] == ["t1", "t2"]

    def test_a_blind_contrarian_prompt_naming_the_proposal_is_rejected(self) -> None:
        with pytest.raises(TemplateVariableError) as caught:
            blind(_llm_prompted("{{ case }} {{ proposal }}"))

        assert caught.value.allowed == frozenset({"case"})


@pytest.mark.unit
class TestFactoredHearing:
    @pytest.mark.asyncio
    async def test_evidence_keeps_challenge_order_whatever_the_finish_order(self) -> None:
        delays = {"a": 0.03, "b": 0.02, "c": 0.01}

        async def slower_first_witness(brief: WitnessBrief[str, str]) -> str:
            await asyncio.sleep(delays[brief.challenge])
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b", "c"), witness=slower_first_witness)
        transcript = await toy_cove(procedure=procedure).run("q")

        assert [outcome.challenge for outcome in transcript.hearings[0].evidence.outcomes] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_unanswered_challenges_within_the_quorum_reach_the_judge(self) -> None:
        async def unsure_witness(brief: WitnessBrief[str, str]) -> str:
            if brief.challenge == "c":
                msg = "no record of c"
                raise LookupError(msg)
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b", "c", "d"), witness=unsure_witness)
        transcript = await toy_cove(procedure=procedure, limits=Limits(quorum=0.75)).run("q")

        evidence = transcript.hearings[0].evidence
        assert evidence.outcomes[2] == Unanswered(
            challenge="c", reason=UnansweredReason.ERROR, error_type="LookupError", message="no record of c"
        )
        assert [item.challenge for item in evidence.answered] == ["a", "b", "d"]
        assert transcript.ruling.endswith("missing [c]")

    @pytest.mark.asyncio
    async def test_losing_the_quorum_dismisses_the_remaining_witnesses(self) -> None:
        async def witness(brief: WitnessBrief[str, str]) -> str:
            if brief.challenge == "a":
                msg = "no record of a"
                raise LookupError(msg)
            await wait_forever()
            return "never"

        procedure = Factored(contrarian=contrarian_raising("a", "b"), witness=witness)

        with pytest.raises(QuorumNotMetError) as caught:
            await toy_cove(procedure=procedure).run("q")

        error = caught.value
        assert (error.total, error.tolerated) == (len(("a", "b")), 0)
        assert [(item.challenge, item.error_type) for item in error.unanswered] == [("a", "LookupError")]
        dismissed = [
            record
            for record in error.record
            if isinstance(record, StepRaised) and record.error_type == "CancelledError"
        ]
        assert [(record.role, record.challenge_index) for record in dismissed] == [(Role.WITNESS, 1)]

    @pytest.mark.asyncio
    async def test_a_witness_past_its_deadline_is_unanswered_for_timeout(self) -> None:
        async def witness(brief: WitnessBrief[str, str]) -> str:
            if brief.challenge == "b":
                await wait_forever()
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b"), witness=witness)
        limits = Limits(witness_timeout_s=0.05, quorum=0.5)

        transcript = await toy_cove(procedure=procedure, limits=limits).run("q")

        (unanswered,) = transcript.hearings[0].evidence.unanswered
        assert unanswered.reason is UnansweredReason.TIMEOUT
        assert unanswered.error_type == "StepTimeoutError"

    @pytest.mark.parametrize(
        "challenges",
        [pytest.param((), id="none"), pytest.param(tuple(f"c{index}" for index in range(13)), id="too-many")],
    )
    @pytest.mark.asyncio
    async def test_the_number_of_challenges_is_bounded(self, challenges: tuple[str, ...]) -> None:
        procedure = Factored(contrarian=contrarian_raising(*challenges), witness=honest_witness)

        with pytest.raises(ChallengeCountError) as caught:
            await toy_cove(procedure=procedure).run("q")

        assert caught.value.count == len(challenges)
        assert (caught.value.minimum, caught.value.maximum) == (Limits().min_challenges, Limits().max_challenges)

    @pytest.mark.asyncio
    async def test_zero_minimum_challenges_allows_a_vacuous_hearing(self) -> None:
        procedure = Factored(contrarian=contrarian_raising(), witness=honest_witness)

        transcript = await toy_cove(procedure=procedure, limits=Limits(min_challenges=0)).run("q")

        assert transcript.hearings[0].evidence.outcomes == ()


@pytest.mark.unit
class TestTwoStepHearing:
    @pytest.mark.asyncio
    async def test_the_panel_answers_every_challenge_in_one_call(self) -> None:
        briefs: list[PanelBrief[str, str]] = []

        async def panel(brief: PanelBrief[str, str]) -> list[str]:
            briefs.append(brief)
            return [f"panel on {challenge}" for challenge in brief.challenges]

        procedure = TwoStep(contrarian=contrarian_raising("a", "b", "c"), panel=panel)
        transcript = await toy_cove(procedure=procedure).run("q")

        assert briefs == [PanelBrief(case="q", challenges=("a", "b", "c"))]
        assert [item.testimony for item in transcript.hearings[0].evidence.answered] == [
            "panel on a",
            "panel on b",
            "panel on c",
        ]
        assert [record.role for record in transcript.record].count(Role.PANEL) == 1

    @pytest.mark.asyncio
    async def test_a_panel_that_miscounts_fails_the_hearing(self) -> None:
        async def forgetful_panel(brief: PanelBrief[str, str]) -> list[str]:
            return [f"panel on {challenge}" for challenge in brief.challenges[:-1]]

        procedure = TwoStep(contrarian=contrarian_raising("a", "b", "c"), panel=forgetful_panel)

        with pytest.raises(PanelMismatchError) as caught:
            await toy_cove(procedure=procedure).run("q")

        assert (caught.value.challenges, caught.value.testimonies) == (3, 2)


@pytest.mark.unit
class TestJointHearing:
    @pytest.mark.asyncio
    async def test_the_inquisitor_challenges_and_answers_in_one_call(self) -> None:
        async def inquisitor(brief: ContrarianBrief[str, str]) -> list[Answered[str, str]]:
            return [Answered(challenge="a", testimony=f"as {brief.proposal} says")]

        transcript = await toy_cove(procedure=Joint(inquisitor=inquisitor)).run("q")

        assert transcript.hearings[0].evidence.outcomes == (
            Answered(challenge="a", testimony="as proposal for q says"),
        )
        roles = [record.role for record in transcript.record]
        assert Role.INQUISITOR in roles
        assert Role.WITNESS not in roles

    @pytest.mark.asyncio
    async def test_the_inquisitor_is_bounded_like_a_contrarian(self) -> None:
        async def silent_inquisitor(brief: ContrarianBrief[str, str]) -> list[Answered[str, str]]:
            del brief
            return []

        with pytest.raises(ChallengeCountError):
            await toy_cove(procedure=Joint(inquisitor=silent_inquisitor)).run("q")


@pytest.mark.unit
class TestFactorReviseHearing:
    @pytest.mark.asyncio
    async def test_each_testimony_is_cross_examined_against_the_proposal(self) -> None:
        briefs: list[CrossExaminationBrief[str, str, str, str]] = []

        async def cross_examiner(brief: CrossExaminationBrief[str, str, str, str]) -> bool:
            briefs.append(brief)
            return brief.challenge in brief.proposal

        async def judge(brief: JudgeBrief[str, str, str, CrossExamined[str, bool]]) -> str:
            contradicted = [item.challenge for item in brief.evidence.answered if not item.testimony.finding]
            return f"contradicted: {contradicted}"

        cove = ChainOfVerification(
            proponent=propose,
            procedure=FactorRevise(
                contrarian=contrarian_raising("proposal", "missing"),
                witness=honest_witness,
                cross_examiner=cross_examiner,
            ),
            judge=judge,
        )

        transcript = await cove.run("q")

        assert transcript.ruling == "contradicted: ['missing']"
        assert transcript.hearings[0].evidence.answered[0].testimony == CrossExamined(
            statement="answer to proposal", finding=True
        )
        assert {brief.proposal for brief in briefs} == {"proposal for q"}


@pytest.mark.unit
class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelling_a_run_propagates_and_records_the_dismissal(self) -> None:
        events: list[CoVeEvent] = []
        testifying = asyncio.Event()

        async def stalling_witness(brief: WitnessBrief[str, str]) -> str:
            testifying.set()
            await wait_forever()
            return brief.challenge

        procedure = Factored(contrarian=contrarian_raising("a"), witness=stalling_witness)
        run = asyncio.create_task(toy_cove(procedure=procedure, observers=[events.append]).run("q"))
        await testifying.wait()

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert any(
            isinstance(event, StepRaised) and event.role is Role.WITNESS and event.error_type == "CancelledError"
            for event in events
        )


@dataclass(frozen=True, slots=True)
class Corroborated:
    """A custom procedure: every challenge is put to two independent witnesses."""

    contrarian: Step[ContrarianBrief[str, str], Sequence[str]]
    first: Step[WitnessBrief[str, str], str]
    second: Step[WitnessBrief[str, str], str]

    async def examine(self, court: Court, *, case: str, proposal: str) -> Evidence[str, tuple[str, str]]:
        challenges = court.admit(
            await court.call(self.contrarian, ContrarianBrief(case=case, proposal=proposal), role=Role.CONTRARIAN)
        )

        async def corroborate(stand: Stand, challenge: str) -> tuple[str, str]:
            brief = WitnessBrief(case=case, challenge=challenge)
            first = await stand.call(self.first, brief, role=Role.WITNESS)
            second = await stand.call(self.second, brief, role=Role.WITNESS)
            return (f"{first} @{stand.challenge_index}", f"{second} @{court.hearing}")

        return await court.hear_each(challenges, corroborate)


@pytest.mark.unit
class TestCustomProcedure:
    @pytest.mark.asyncio
    async def test_a_custom_procedure_gets_the_engine_bounds_and_record(self) -> None:
        async def second_opinion(brief: WitnessBrief[str, str]) -> str:
            return f"also {brief.challenge}"

        async def judge(brief: JudgeBrief[str, str, str, tuple[str, str]]) -> str:
            return "; ".join(" & ".join(item.testimony) for item in brief.evidence.answered)

        cove = ChainOfVerification(
            proponent=propose,
            procedure=Corroborated(
                contrarian=contrarian_raising("a", "b"), first=honest_witness, second=second_opinion
            ),
            judge=judge,
        )

        transcript = await cove.run("q")

        assert transcript.ruling == "answer to a @0 & also a @1; answer to b @1 & also b @1"
        witness_calls = [record.challenge_index for record in transcript.record if record.role is Role.WITNESS]
        assert sorted(index for index in witness_calls if index is not None) == [0, 0, 1, 1]
