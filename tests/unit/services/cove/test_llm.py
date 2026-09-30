"""The LLM layer: llm_step, JinjaPrompt, the port decorators, and the scripted test double."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from omniadapters.services.cove import (
    Budgeted,
    BudgetExceededError,
    CallBudget,
    Factored,
    JinjaPrompt,
    Limits,
    QuorumNotMetError,
    Role,
    StepFailedError,
    Throttled,
    WitnessBrief,
    llm_step,
)
from omniadapters.services.cove.testing import ScriptedLLM
from tests.unit.services.cove._toy import InFlight, contrarian_raising, toy_cove

if TYPE_CHECKING:
    from pathlib import Path

    from omniadapters.services.cove import Messages
    from omniadapters.services.cove.testing import RecordedCall


class _Reply(BaseModel):
    text: str


class _Other(BaseModel):
    value: int


@dataclass(frozen=True, slots=True)
class _Brief:
    case: str
    challenge: str


def _echo(call: RecordedCall) -> BaseModel:
    return _Reply(text=call.text.upper())


class _SlowLLM:
    """A StructuredLLM whose every call takes a little while, counting how many overlap."""

    def __init__(self, in_flight: InFlight) -> None:
        self._in_flight = in_flight

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        await self._in_flight.hold(str(len(messages)))
        return response_model.model_validate({"text": "slow"})


@pytest.mark.unit
class TestLLMStep:
    @pytest.mark.asyncio
    async def test_renders_the_brief_calls_the_model_and_parses(self) -> None:
        llm = ScriptedLLM(_echo)
        step = llm_step(
            llm=llm,
            prompt=JinjaPrompt(system="You check {{ case }}.", user="{{ challenge }}?"),
            response_model=_Reply,
            parse=lambda reply: reply.text,
        )

        result = await step(_Brief(case="capitals", challenge="Canberra"))

        assert result == "YOU CHECK CAPITALS.\nCANBERRA?"
        (call,) = llm.calls
        assert call.response_model is _Reply
        assert [message["role"] for message in call.messages] == ["system", "user"]

    @pytest.mark.asyncio
    async def test_without_parse_returns_the_response(self) -> None:
        step = llm_step(
            llm=ScriptedLLM(_echo), prompt=JinjaPrompt(system="s", user="{{ case }}"), response_model=_Reply
        )

        assert await step(_Brief(case="x", challenge="y")) == _Reply(text="S\nX")

    def test_reports_template_variables_only_for_jinja_prompts(self) -> None:
        def typed_prompt(brief: _Brief) -> Messages:
            return [{"role": "user", "content": brief.challenge}]

        jinja = llm_step(
            llm=ScriptedLLM(_echo), prompt=JinjaPrompt(system="s", user="{{ case }}"), response_model=_Reply
        )
        typed = llm_step(llm=ScriptedLLM(_echo), prompt=typed_prompt, response_model=_Reply)

        assert jinja.template_variables == frozenset({"case"})
        assert typed.template_variables is None


@pytest.mark.unit
class TestJinjaPrompt:
    def test_collects_top_level_names_but_not_loop_variables(self) -> None:
        prompt = JinjaPrompt(
            system="{{ case.text }}",
            user="{% for item in evidence.answered %}{{ loop.index }} {{ item.challenge }}{% endfor %}",
        )

        assert prompt.variables == frozenset({"case", "evidence"})

    def test_an_undefined_name_raises_instead_of_rendering_blank(self) -> None:
        prompt = JinjaPrompt(system="s", user="{{ case }} {{ proposal }}")

        with pytest.raises(Exception, match="proposal"):
            prompt(_Brief(case="x", challenge="y"))

    def test_renders_pydantic_model_briefs(self) -> None:
        messages = JinjaPrompt(system="s", user="{{ text }}")(_Reply(text="hello"))

        assert messages[1] == {"role": "user", "content": "hello"}

    def test_rejects_briefs_that_are_not_records(self) -> None:
        with pytest.raises(TypeError, match="dataclass or pydantic model"):
            JinjaPrompt(system="s", user="u")("not a brief")

    def test_loads_templates_from_files(self, tmp_path: Path) -> None:
        (tmp_path / "system.j2").write_text("Check {{ case }}.", encoding="utf-8")
        (tmp_path / "user.j2").write_text("{{ challenge }}", encoding="utf-8")

        prompt = JinjaPrompt.from_files(tmp_path, system="system.j2", user="user.j2")

        assert prompt(_Brief(case="facts", challenge="why?")) == [
            {"role": "system", "content": "Check facts."},
            {"role": "user", "content": "why?"},
        ]


@pytest.mark.unit
class TestThrottled:
    @pytest.mark.asyncio
    async def test_one_limiter_caps_calls_across_models(self) -> None:
        in_flight = InFlight()
        slots = 2
        limiter = asyncio.Semaphore(slots)
        first = Throttled(inner=_SlowLLM(in_flight), limiter=limiter)
        second = Throttled(inner=_SlowLLM(in_flight), limiter=limiter)

        await asyncio.gather(*(llm.acreate([], _Reply) for llm in (first, second) for _ in range(4)))

        assert in_flight.peak == slots


@pytest.mark.unit
class TestBudget:
    @pytest.mark.asyncio
    async def test_refuses_calls_once_the_budget_is_spent(self) -> None:
        inner = ScriptedLLM(_echo)
        budget = CallBudget(max_calls=2)
        llm = Budgeted(inner=inner, budget=budget)

        await llm.acreate([], _Reply)
        await llm.acreate([], _Reply)
        with pytest.raises(BudgetExceededError) as caught:
            await llm.acreate([], _Reply)

        assert len(inner.calls) == budget.max_calls
        assert budget.spent == budget.max_calls
        assert caught.value.context == {"max_calls": budget.max_calls}

    def test_needs_at_least_one_call(self) -> None:
        with pytest.raises(ValueError, match="at least one call"):
            CallBudget(max_calls=0)

    @pytest.mark.asyncio
    async def test_a_spent_budget_ends_the_run(self) -> None:
        llm = Budgeted(inner=ScriptedLLM(_echo), budget=CallBudget(max_calls=1))
        witness = llm_step(
            llm=llm, prompt=JinjaPrompt(system="s", user="{{ challenge }}"), response_model=_Reply, parse=str
        )
        procedure = Factored(contrarian=contrarian_raising("a", "b", "c"), witness=witness)

        with pytest.raises(QuorumNotMetError) as caught:
            await toy_cove(procedure=procedure).run("q")

        assert {item.error_type for item in caught.value.unanswered} == {"BudgetExceededError"}

    @pytest.mark.asyncio
    async def test_a_spent_budget_fails_a_whole_hearing_step(self) -> None:
        llm = Budgeted(inner=ScriptedLLM(_echo), budget=CallBudget(max_calls=1))
        judge = llm_step(
            llm=llm, prompt=JinjaPrompt(system="s", user="{{ proposal }}"), response_model=_Reply, parse=str
        )
        await llm.acreate([], _Reply)

        with pytest.raises(StepFailedError) as caught:
            await toy_cove(judge=judge, limits=Limits()).run("q")

        assert caught.value.role is Role.JUDGE
        assert isinstance(caught.value.__cause__, BudgetExceededError)


@pytest.mark.unit
class TestScriptedLLM:
    @pytest.mark.asyncio
    async def test_rejects_a_reply_of_the_wrong_model(self) -> None:
        llm = ScriptedLLM(lambda _: _Other(value=1))

        with pytest.raises(TypeError, match="answered a _Reply request with a _Other"):
            await llm.acreate([{"role": "user", "content": "hi"}], _Reply)

        assert [call.text for call in llm.calls] == ["hi"]

    @pytest.mark.asyncio
    async def test_is_usable_as_a_witness_through_llm_step(self) -> None:
        witness = llm_step(
            llm=ScriptedLLM(_echo),
            prompt=JinjaPrompt(system="witness", user="{{ challenge }}"),
            response_model=_Reply,
            parse=lambda reply: reply.text,
        )

        answer = await witness(WitnessBrief(case="q", challenge="why"))

        assert answer == "WITNESS\nWHY"
