"""The factual-QA recipe: ``playground/critic`` as a chain-of-verification recipe, with critic's defects fixed."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from omniadapters.services.cove import ContrarianBrief, Limits, Messages, WitnessBrief
from omniadapters.services.cove.backends import PydanticAIStructuredLLM
from omniadapters.services.cove.recipes.factual_qa import (
    DEFAULT_PROMPTS,
    MAX_CHALLENGES,
    MIN_CHALLENGES,
    Answer,
    Challenges,
    Consistency,
    FactualQAConfig,
    FactualQAProcedure,
    FactualQARoles,
    Inquiry,
    InquiryItem,
    PanelAnswers,
    Question,
    Verdict,
    VerifiedAnswer,
    WitnessAnswer,
    build_cross_examined_factual_qa,
    build_factual_qa,
    open_factual_qa,
)
from omniadapters.services.cove.testing import RecordedCall, ScriptedLLM

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models.function import AgentInfo

_QUESTION: Final = Question(text="Who was the first woman to win two Nobel Prizes, and in which years?")
_PROPOSAL: Final = "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908."
_CORRECTED: Final = "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911."
_HALLUCINATED_YEAR: Final = "1908"
_TRUE_YEAR: Final = "1911"
_QUESTIONS: Final = (
    "In which year did Marie Curie win the Nobel Prize in Physics?",
    "In which year did Marie Curie win the Nobel Prize in Chemistry?",
    "Who was the first woman to win two Nobel Prizes?",
)


def _witness_answer(question: str) -> WitnessAnswer:
    if "Physics" in question:
        return WitnessAnswer(answer="1903", explanation="The Physics prize was shared with Pierre Curie.")
    if "Chemistry" in question:
        return WitnessAnswer(answer=_TRUE_YEAR, explanation="The Chemistry prize was awarded to her alone.")
    return WitnessAnswer(answer="Marie Curie", explanation="She won in both Physics and Chemistry.")


def _courtroom(call: RecordedCall) -> BaseModel:  # noqa: PLR0911 - one reply per response model
    """Answer every role in a scripted world where the proponent misremembers a year the witnesses know."""
    if call.response_model is Answer:
        return Answer(text=_PROPOSAL)
    if call.response_model is Challenges:
        return Challenges(questions=list(_QUESTIONS))
    if call.response_model is WitnessAnswer:
        return _witness_answer(call.text)
    if call.response_model is PanelAnswers:
        return PanelAnswers(answers=[_witness_answer(question) for question in _QUESTIONS])
    if call.response_model is Inquiry:
        # The joint inquisitor answers while reading the proposal, and repeats its error.
        return Inquiry(
            items=[
                InquiryItem(
                    question=question,
                    answer=_HALLUCINATED_YEAR if "Chemistry" in question else _witness_answer(question).answer,
                    explanation="As the proposed answer says.",
                )
                for question in _QUESTIONS
            ]
        )
    if call.response_model is Consistency:
        contradicted = _TRUE_YEAR in call.text
        return Consistency(consistent=not contradicted, note="Compared the independent answer with the proposal.")
    if call.response_model is VerifiedAnswer:
        revised = _TRUE_YEAR in call.text
        return VerifiedAnswer(
            answer=_CORRECTED if revised else _PROPOSAL,
            verdict=Verdict.REVISED if revised else Verdict.CONFIRMED,
            confidence=0.9 if revised else 0.6,
            rationale="Decided by the evidence.",
        )
    raise AssertionError(call.response_model.__name__)


@dataclass
class _Cast:
    """One scripted LLM per role, so each role's prompts can be inspected on their own."""

    proponent: ScriptedLLM = field(default_factory=lambda: ScriptedLLM(_courtroom))
    contrarian: ScriptedLLM = field(default_factory=lambda: ScriptedLLM(_courtroom))
    witness: ScriptedLLM = field(default_factory=lambda: ScriptedLLM(_courtroom))
    judge: ScriptedLLM = field(default_factory=lambda: ScriptedLLM(_courtroom))

    def roles(self) -> FactualQARoles:
        return FactualQARoles(
            proponent=self.proponent, contrarian=self.contrarian, witness=self.witness, judge=self.judge
        )


@pytest.mark.unit
class TestProcedures:
    @pytest.mark.parametrize("procedure", [FactualQAProcedure.FACTORED, FactualQAProcedure.TWO_STEP])
    @pytest.mark.asyncio
    async def test_independent_witnesses_correct_the_proposal(self, procedure: FactualQAProcedure) -> None:
        cast = _Cast()

        transcript = await build_factual_qa(cast.roles(), procedure=procedure).run(_QUESTION)

        assert transcript.ruling.verdict is Verdict.REVISED
        assert transcript.ruling.answer == _CORRECTED
        assert cast.witness.calls
        assert all(_HALLUCINATED_YEAR not in call.text for call in cast.witness.calls)

    @pytest.mark.asyncio
    async def test_factored_puts_each_question_to_its_own_witness(self) -> None:
        cast = _Cast()

        await build_factual_qa(cast.roles()).run(_QUESTION)

        assert {call.messages.user for call in cast.witness.calls} == set(_QUESTIONS)

    @pytest.mark.asyncio
    async def test_the_joint_variant_repeats_the_proposals_error(self) -> None:
        cast = _Cast()

        transcript = await build_factual_qa(cast.roles(), procedure=FactualQAProcedure.JOINT).run(_QUESTION)

        assert transcript.ruling.verdict is Verdict.CONFIRMED
        assert transcript.ruling.answer == _PROPOSAL
        assert not cast.witness.calls
        (inquiry,) = [call for call in cast.contrarian.calls if call.response_model is Inquiry]
        assert _HALLUCINATED_YEAR in inquiry.text

    @pytest.mark.asyncio
    async def test_cross_examination_flags_the_contradiction(self) -> None:
        cast = _Cast()
        cross_examiner = ScriptedLLM(_courtroom)

        transcript = await build_cross_examined_factual_qa(cast.roles(), cross_examiner=cross_examiner).run(_QUESTION)

        findings = [item.testimony.finding.consistent for item in transcript.hearings[0].evidence.answered]
        assert findings == [True, False, True]
        assert transcript.ruling.answer == _CORRECTED
        assert all(_HALLUCINATED_YEAR in call.text for call in cross_examiner.calls)
        assert all(_HALLUCINATED_YEAR not in call.text for call in cast.witness.calls)


@pytest.mark.unit
class TestCriticDefectsStayFixed:
    @pytest.mark.asyncio
    async def test_the_judge_rules_on_evidence_not_on_a_self_assessment(self) -> None:
        cast = _Cast()

        await build_factual_qa(cast.roles()).run(_QUESTION)

        (judge_call,) = cast.judge.calls
        assert f"Proposed answer: {_PROPOSAL}" in judge_call.text
        assert f"A: {_TRUE_YEAR}" in judge_call.text
        assert "appears" not in judge_call.text

    def test_witness_answers_are_open_not_yes_or_no(self) -> None:
        answer = WitnessAnswer.model_json_schema()["properties"]["answer"]

        assert answer["type"] == "string"
        assert "enum" not in answer

    def test_the_ruling_carries_the_final_answer(self) -> None:
        assert "answer" in VerifiedAnswer.model_fields

    def test_the_question_bounds_have_one_owner(self) -> None:
        questions = Challenges.model_json_schema()["properties"]["questions"]
        brief = ContrarianBrief(case=_QUESTION, proposal=Answer(text=_PROPOSAL))

        system = DEFAULT_PROMPTS.contrarian(brief).system

        assert (questions["minItems"], questions["maxItems"]) == (MIN_CHALLENGES, MAX_CHALLENGES)
        assert f"{MIN_CHALLENGES} to {MAX_CHALLENGES}" in system

    @pytest.mark.asyncio
    async def test_a_prompt_can_be_replaced_by_a_typed_function(self) -> None:
        def terse_witness(brief: WitnessBrief[Question, str]) -> Messages:
            return Messages(system="Answer in one word.", user=brief.challenge)

        cast = _Cast()

        await build_factual_qa(cast.roles(), prompts=dataclasses.replace(DEFAULT_PROMPTS, witness=terse_witness)).run(
            _QUESTION
        )

        assert {call.messages.system for call in cast.witness.calls} == {"Answer in one word."}


_RESPONSE_MODELS: Final = {
    model.__name__: model
    for model in (Answer, Challenges, WitnessAnswer, PanelAnswers, Inquiry, Consistency, VerifiedAnswer)
}


def _courtroom_on_pydantic_ai(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Play the scripted courtroom as a pydantic-ai model, answering the requested schema with its output tool."""
    (output_tool,) = info.output_tools
    prompt = "".join(
        part.content
        for part in messages[-1].parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )
    call = RecordedCall(
        messages=Messages(system=info.instructions or "", user=prompt),
        response_model=_RESPONSE_MODELS[output_tool.parameters_json_schema["title"]],
    )
    return ModelResponse(parts=[ToolCallPart(output_tool.name, _courtroom(call).model_dump(mode="json"))])


def _pydantic_ai_cast() -> FactualQARoles:
    llm = PydanticAIStructuredLLM(Agent(FunctionModel(_courtroom_on_pydantic_ai)))
    return FactualQARoles(proponent=llm, contrarian=llm, witness=llm, judge=llm)


@pytest.mark.unit
class TestOnPydanticAI:
    @pytest.mark.parametrize(
        ("procedure", "final_answer"),
        [
            (FactualQAProcedure.FACTORED, _CORRECTED),
            (FactualQAProcedure.TWO_STEP, _CORRECTED),
            (FactualQAProcedure.JOINT, _PROPOSAL),
        ],
    )
    @pytest.mark.asyncio
    async def test_every_procedure_rules_as_it_does_on_the_scripted_cast(
        self, procedure: FactualQAProcedure, final_answer: str
    ) -> None:
        transcript = await build_factual_qa(_pydantic_ai_cast(), procedure=procedure).run(_QUESTION)

        assert transcript.ruling.answer == final_answer

    @pytest.mark.asyncio
    async def test_cross_examination_flags_the_contradiction(self) -> None:
        roles = _pydantic_ai_cast()

        transcript = await build_cross_examined_factual_qa(roles, cross_examiner=roles.witness).run(_QUESTION)

        findings = [item.testimony.finding.consistent for item in transcript.hearings[0].evidence.answered]
        assert findings == [True, False, True]
        assert transcript.ruling.answer == _CORRECTED


def _role(provider: str, model_name: str) -> dict[str, object]:
    return {
        "provider_config": {"provider": provider, "api_key": "test"},
        "model_name": model_name,
        "model_settings": {"temperature": 0.0},
    }


_ROLES: Final = {
    "proponent": {**_role("google", "gemini-2.5-flash"), "output_mode": "native"},
    "contrarian": _role("openai", "gpt-4o-mini"),
    "witness": _role("openai", "gpt-4o-mini"),
    "judge": _role("anthropic", "claude-sonnet-5-5"),
}


@pytest.mark.unit
class TestFactualQAConfig:
    def test_parses_a_yaml_shaped_mapping(self) -> None:
        config = FactualQAConfig.model_validate(
            {**_ROLES, "procedure": "two_step", "limits": {"max_concurrent_steps": 4, "quorum": 0.75}}
        )

        assert config.procedure is FactualQAProcedure.TWO_STEP
        assert config.limits == Limits(max_concurrent_steps=4, quorum=0.75)

    def test_defaults_to_the_factored_procedure_and_default_limits(self) -> None:
        config = FactualQAConfig.model_validate(_ROLES)

        assert config.procedure is FactualQAProcedure.FACTORED
        assert config.limits == Limits()

    def test_rejects_critics_old_role_names(self) -> None:
        with pytest.raises(ValidationError, match="drafter"):
            FactualQAConfig.model_validate({**_ROLES, "drafter": _ROLES["proponent"]})

    @pytest.mark.asyncio
    async def test_opens_one_client_per_role_and_builds_the_recipe(self) -> None:
        config = FactualQAConfig.model_validate({**_ROLES, "limits": {"max_concurrent_steps": 4}})

        async with open_factual_qa(config) as cove:
            assert cove.limits.max_concurrent_steps == config.limits.max_concurrent_steps
