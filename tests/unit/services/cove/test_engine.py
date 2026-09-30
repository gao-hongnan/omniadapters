"""The engine owns the proceedings: hearings, retrials, batches, observers, bounds and the transcript."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from omniadapters.services.cove import (
    Answered,
    CaseBrief,
    ChainOfVerification,
    CoVeEvent,
    CoVeRunError,
    Disposition,
    Factored,
    JinjaPrompt,
    JudgeBrief,
    Limits,
    Reopen,
    Retrial,
    Role,
    StepFailedError,
    StepTimeoutError,
    TemplateVariableError,
    WitnessBrief,
    llm_step,
    logging_observer,
)
from omniadapters.services.cove.events import RunFailed, RunFinished, RunStarted, StepCompleted, StepRaised
from omniadapters.services.cove.testing import ScriptedLLM
from tests.unit.services.cove._toy import (
    InFlight,
    contrarian_raising,
    honest_witness,
    propose,
    toy_cove,
    wait_forever,
)

if TYPE_CHECKING:
    from omniadapters.services.cove.testing import RecordedCall


class _Ruling(BaseModel):
    text: str


@pytest.mark.unit
class TestSingleHearing:
    @pytest.mark.asyncio
    async def test_rules_on_the_proposal_with_evidence_in_challenge_order(self) -> None:
        transcript = await toy_cove().run("q")

        assert transcript.case == "q"
        assert transcript.disposition is Disposition.SINGLE_HEARING
        assert len(transcript.hearings) == 1
        hearing = transcript.hearings[0]
        assert hearing.number == 1
        assert hearing.proposal == "proposal for q"
        assert hearing.evidence.outcomes == (
            Answered(challenge="a", testimony="answer to a"),
            Answered(challenge="b", testimony="answer to b"),
            Answered(challenge="c", testimony="answer to c"),
        )
        assert transcript.ruling == (
            "ruling on proposal for q [a=answer to a, b=answer to b, c=answer to c] missing []"
        )

    @pytest.mark.asyncio
    async def test_records_every_step_under_one_run_id(self) -> None:
        transcript = await toy_cove().run("q")

        assert {record.run_id for record in transcript.record} == {transcript.run_id}
        assert all(isinstance(record, StepCompleted) for record in transcript.record)
        roles = [record.role for record in transcript.record]
        assert roles[:2] == [Role.PROPONENT, Role.CONTRARIAN]
        assert roles[-1] is Role.JUDGE
        witness_indexes = sorted(
            record.challenge_index
            for record in transcript.record
            if record.role is Role.WITNESS and record.challenge_index is not None
        )
        assert witness_indexes == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_transcript_serialises_to_self_describing_json(self) -> None:
        transcript = await toy_cove().run("q")

        document = json.loads(transcript.to_json())

        assert set(document) == {"run_id", "case", "hearings", "disposition", "record"}
        assert document["disposition"] == "single_hearing"
        assert {outcome["kind"] for outcome in document["hearings"][0]["evidence"]["outcomes"]} == {"answered"}
        assert {record["kind"] for record in document["record"]} == {"step_completed"}


@pytest.mark.unit
class TestFailures:
    @pytest.mark.asyncio
    async def test_proponent_failure_is_a_step_failure_with_the_partial_record(self) -> None:
        async def broken_proponent(brief: CaseBrief[str]) -> str:
            msg = f"cannot draft {brief.case}"
            raise ValueError(msg)

        with pytest.raises(StepFailedError) as caught:
            await toy_cove(proponent=broken_proponent).run("q")

        error = caught.value
        assert error.role is Role.PROPONENT
        assert error.hearing == 1
        assert isinstance(error.__cause__, ValueError)
        assert "cannot draft q" in str(error)
        assert [type(record) for record in error.record] == [StepRaised]
        assert error.context["role"] is Role.PROPONENT

    @pytest.mark.asyncio
    async def test_a_judge_past_its_deadline_fails_the_run(self) -> None:
        async def stalling_judge(brief: JudgeBrief[str, str, str, str]) -> str:
            del brief
            await wait_forever()
            return "never"

        cove = toy_cove(judge=stalling_judge, limits=Limits(timeout_s=0.05))

        with pytest.raises(StepFailedError) as caught:
            await cove.run("q")

        assert caught.value.role is Role.JUDGE
        assert isinstance(caught.value.__cause__, StepTimeoutError)
        assert isinstance(caught.value.__cause__, TimeoutError)


@pytest.mark.unit
class TestRetrial:
    @pytest.mark.asyncio
    async def test_reopens_until_the_ruling_stands(self) -> None:
        def reconsider(ruling: str) -> Reopen[str] | None:
            return None if "revised" in ruling else Reopen(proposal="revised proposal")

        transcript = await toy_cove(retrial=Retrial(max_hearings=3, reconsider=reconsider)).run("q")

        assert transcript.disposition is Disposition.SETTLED
        assert [hearing.proposal for hearing in transcript.hearings] == ["proposal for q", "revised proposal"]
        assert [hearing.number for hearing in transcript.hearings] == [1, 2]

    @pytest.mark.asyncio
    async def test_stops_at_the_hearing_limit(self) -> None:
        def always_reopen(ruling: str) -> Reopen[str]:
            return Reopen(proposal=f"again after {len(ruling)}")

        retrial = Retrial(max_hearings=3, reconsider=always_reopen)

        transcript = await toy_cove(retrial=retrial).run("q")

        assert transcript.disposition is Disposition.HEARING_LIMIT
        assert len(transcript.hearings) == retrial.max_hearings

    def test_a_retrial_needs_at_least_two_hearings(self) -> None:
        with pytest.raises(ValueError, match="max_hearings >= 2"):
            Retrial[str, str](max_hearings=1, reconsider=lambda _: None)


@pytest.mark.unit
class TestBatches:
    @pytest.mark.asyncio
    async def test_failed_cases_come_back_in_place(self) -> None:
        async def picky_proponent(brief: CaseBrief[str]) -> str:
            if brief.case == "boom":
                msg = "unanswerable"
                raise ValueError(msg)
            return await propose(brief)

        results = await toy_cove(proponent=picky_proponent).run_many(["a", "boom", "c"])

        assert [type(result).__name__ for result in results] == ["Transcript", "StepFailedError", "Transcript"]
        failure = results[1]
        assert isinstance(failure, CoVeRunError)
        assert failure.record

    @pytest.mark.asyncio
    async def test_caps_concurrent_steps_across_runs(self) -> None:
        in_flight = InFlight()

        async def slow_witness(brief: WitnessBrief[str, str]) -> str:
            await in_flight.hold(brief.challenge)
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b", "c"), witness=slow_witness)
        limits = Limits(max_concurrent_steps=2, max_concurrent_runs=5)

        results = await toy_cove(procedure=procedure, limits=limits).run_many([f"case {i}" for i in range(5)])

        assert all(not isinstance(result, CoVeRunError) for result in results)
        assert in_flight.peak == limits.max_concurrent_steps

    @pytest.mark.asyncio
    async def test_caps_concurrent_runs(self) -> None:
        runs = InFlight()

        async def slow_proponent(brief: CaseBrief[str]) -> str:
            await runs.hold(brief.case)
            return await propose(brief)

        limits = Limits(max_concurrent_steps=64, max_concurrent_runs=2)
        cases = [f"case {index}" for index in range(6)]

        results = await toy_cove(proponent=slow_proponent, limits=limits).run_many(cases)

        assert len(results) == len(cases)
        assert runs.peak == limits.max_concurrent_runs
        assert runs.started == cases

    @pytest.mark.asyncio
    async def test_an_empty_batch_is_empty(self) -> None:
        assert await toy_cove().run_many([]) == []

    def test_an_engine_is_reusable_across_event_loops(self) -> None:
        cove = toy_cove()

        first = asyncio.run(cove.run("q"))
        second = asyncio.run(cove.run("q"))

        assert first.ruling == second.ruling
        assert first.run_id != second.run_id


@pytest.mark.unit
class TestObservers:
    @pytest.mark.asyncio
    async def test_receive_the_run_from_start_to_finish(self) -> None:
        events: list[CoVeEvent] = []

        transcript = await toy_cove(observers=[events.append]).run("q")

        assert isinstance(events[0], RunStarted)
        assert isinstance(events[-1], RunFinished)
        assert events[-1].disposition is Disposition.SINGLE_HEARING
        assert all(event.run_id == transcript.run_id for event in events)
        assert sum(isinstance(event, StepCompleted) for event in events) == len(transcript.record)

    @pytest.mark.asyncio
    async def test_an_observer_that_raises_is_logged_and_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        def broken_observer(event: CoVeEvent) -> None:
            msg = f"cannot render {event.kind}"
            raise RuntimeError(msg)

        with caplog.at_level(logging.ERROR, logger="omniadapters.services.cove"):
            transcript = await toy_cove(observers=[broken_observer]).run("q")

        assert transcript.disposition is Disposition.SINGLE_HEARING
        assert any("observer" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_failed_run_ends_with_run_failed(self) -> None:
        events: list[CoVeEvent] = []

        async def broken_proponent(brief: CaseBrief[str]) -> str:
            raise ValueError(brief.case)

        with pytest.raises(StepFailedError):
            await toy_cove(proponent=broken_proponent, observers=[events.append]).run("q")

        assert isinstance(events[-1], RunFailed)
        assert events[-1].error_type == "StepFailedError"

    @pytest.mark.asyncio
    async def test_logging_observer_logs_every_event(self, caplog: pytest.LogCaptureFixture) -> None:
        async def flaky_witness(brief: WitnessBrief[str, str]) -> str:
            if brief.challenge == "b":
                msg = "no idea"
                raise LookupError(msg)
            return await honest_witness(brief)

        procedure = Factored(contrarian=contrarian_raising("a", "b"), witness=flaky_witness)
        with caplog.at_level(logging.DEBUG, logger="omniadapters.services.cove"):
            await toy_cove(procedure=procedure, limits=Limits(quorum=0.5), observers=[logging_observer]).run("q")
            with pytest.raises(StepFailedError):
                await toy_cove(proponent=_raise_value_error, observers=[logging_observer]).run("q")

        messages = [record.getMessage() for record in caplog.records]
        assert any("started" in message for message in messages)
        assert any("witness[1] raised LookupError" in message for message in messages)
        assert any("finished (single_hearing)" in message for message in messages)
        assert any("failed with StepFailedError" in message for message in messages)


async def _raise_value_error(brief: CaseBrief[str]) -> str:
    raise ValueError(brief.case)


@pytest.mark.unit
class TestBuildTimeChecks:
    def test_a_judge_prompt_naming_a_field_the_brief_lacks_is_rejected(self) -> None:
        def script(call: RecordedCall) -> BaseModel:
            raise AssertionError(call)

        judge = llm_step(
            llm=ScriptedLLM(script),
            prompt=JinjaPrompt(system="judge", user="{{ proposal }} {{ verdict }}"),
            response_model=_Ruling,
        )

        with pytest.raises(TemplateVariableError) as caught:
            ChainOfVerification(
                proponent=propose,
                procedure=Factored(contrarian=contrarian_raising("a"), witness=honest_witness),
                judge=judge,
            )

        assert caught.value.role is Role.JUDGE
        assert caught.value.unknown == frozenset({"verdict"})

    def test_limits_reject_an_inverted_challenge_window(self) -> None:
        with pytest.raises(ValidationError, match="min_challenges"):
            Limits(min_challenges=5, max_challenges=3)

    def test_engine_exposes_its_limits(self) -> None:
        limits = Limits(max_challenges=3)

        assert toy_cove(limits=limits).limits is limits
