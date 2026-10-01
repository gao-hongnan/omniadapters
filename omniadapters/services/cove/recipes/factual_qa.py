"""Factual question answering, verified by chain of verification. This recipe replaces ``playground/critic``.

Critic's drafter, skeptic, fact-checker and judge become the proponent, contrarian, witness and
judge. Critic's defects are fixed along the way:

- Witnesses give **open** answers. Critic forced ``Literal["yes", "no"]`` onto open questions,
  and the paper found yes/no verification weaker because models tend to agree.
- The ruling **carries the corrected answer**. Critic's verdict had no field for it, so the
  paper's final step was skipped.
- The proposal is **only the answer**. Critic's draft carried a self-assessment that anchored the
  later roles.
- The question bounds have **one owner**: :data:`MIN_CHALLENGES` and :data:`MAX_CHALLENGES` feed
  both the schema and the prompt.
- Each role holds **one client**, opened once and closed on exit, instead of one client per call.

From config, each role is a pydantic-ai agent (see :func:`open_factual_qa`); in code, any
:class:`~omniadapters.services.cove.llm.StructuredLLM` can play a role (see :func:`build_factual_qa`).
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Annotated, Final, assert_never

from pydantic import BaseModel, ConfigDict, Field

from ..backends import RoleConfig, open_role
from ..engine import ChainOfVerification
from ..enums import Role
from ..evidence import Answered, CrossExamined
from ..limits import DEFAULT_LIMITS, Limits
from ..llm import llm_step
from ..procedures import Factored, FactorRevise, Joint, TwoStep
from ..prompts import JinjaPrompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from ..briefs import CaseBrief, ContrarianBrief, CrossExaminationBrief, JudgeBrief, PanelBrief, WitnessBrief
    from ..events import Observer
    from ..llm import LLMStep, StructuredLLM
    from ..procedures import Procedure
    from ..prompts import Prompt

MIN_CHALLENGES: Final = 3
MAX_CHALLENGES: Final = 6


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Question(_Schema):
    """The case: a question to answer."""

    text: Annotated[str, Field(min_length=1, description="The question to answer.")]


class Answer(_Schema):
    """The proposal: the proponent's answer, and nothing else, so no self-assessment anchors the judge."""

    text: Annotated[str, Field(description="A direct, complete answer that states the facts it relies on.")]


class Challenges(_Schema):
    """The contrarian's open verification questions."""

    questions: Annotated[
        list[str],
        Field(
            min_length=MIN_CHALLENGES,
            max_length=MAX_CHALLENGES,
            description="Open questions, one per factual claim, each answerable without the proposed answer.",
        ),
    ]


class WitnessAnswer(_Schema):
    """A witness's open answer to one verification question."""

    answer: Annotated[str, Field(description="A short, direct answer to the verification question.")]
    explanation: Annotated[str, Field(description="One or two sentences of supporting evidence.")]


class PanelAnswers(_Schema):
    """The two-step panel's answers, one per question."""

    answers: Annotated[list[WitnessAnswer], Field(description="Exactly one answer per question, in the order asked.")]


class InquiryItem(_Schema):
    """One question the joint-variant inquisitor asked and answered."""

    question: Annotated[str, Field(description="An open verification question about one factual claim.")]
    answer: Annotated[str, Field(description="A short, direct answer to the question.")]
    explanation: Annotated[str, Field(description="One or two sentences of supporting evidence.")]


class Inquiry(_Schema):
    """The joint-variant inquisitor's questions and answers."""

    items: Annotated[list[InquiryItem], Field(min_length=MIN_CHALLENGES, max_length=MAX_CHALLENGES)]


class Consistency(_Schema):
    """The cross-examiner's finding on one independent answer."""

    consistent: Annotated[bool, Field(description="Whether the independent answer agrees with the proposed answer.")]
    note: Annotated[str, Field(description="The specific agreement or contradiction, in one sentence.")]


class Verdict(StrEnum):
    """Whether the evidence let the proposal stand."""

    CONFIRMED = auto()
    REVISED = auto()


class VerifiedAnswer(_Schema):
    """The ruling: the final answer, and why."""

    answer: Annotated[
        str,
        Field(
            description=(
                "The final answer: the proposed answer if the evidence supports it, "
                "otherwise a corrected answer written from the evidence."
            )
        ),
    ]
    verdict: Annotated[
        Verdict, Field(description="`confirmed` if the proposed answer stands, `revised` if the evidence changed it.")
    ]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence in the final answer, from 0 to 1.")]
    rationale: Annotated[str, Field(description="The evidence that decided the verdict.")]


_PROPONENT: Final = JinjaPrompt(
    system=(
        "You are a knowledgeable assistant. Answer the question directly and completely. "
        "State the specific facts your answer relies on: names, dates, numbers, places."
    ),
    user="{{ case.text }}",
)
_CONTRARIAN: Final = JinjaPrompt(
    system=(
        "You are a contrarian fact-checker. The proposed answer may contain factual errors. "
        f"Write {MIN_CHALLENGES} to {MAX_CHALLENGES} open verification questions, one per factual claim "
        "the answer relies on: names, dates, numbers, places, relationships. Each question must be "
        "answerable on its own, without the proposed answer, and must ask for the fact instead of "
        'stating it: ask "When did X happen?", never "Did X happen in 1908?".'
    ),
    user="Question: {{ case.text }}\nProposed answer: {{ proposal.text }}",
)
_WITNESS: Final = JinjaPrompt(
    system=(
        "You are a careful researcher. Answer the question accurately and concisely from your own "
        "knowledge. If you are not sure, say so instead of guessing."
    ),
    user="{{ challenge }}",
)
_PANEL: Final = JinjaPrompt(
    system=(
        "You are a careful researcher. Answer each question independently, accurately and concisely. "
        "Return exactly one answer per question, in the order asked. If you are not sure of an answer, "
        "say so instead of guessing."
    ),
    user="{% for question in challenges %}{{ loop.index }}. {{ question }}\n{% endfor %}",
)
_INQUISITOR: Final = JinjaPrompt(
    system=(
        f"You are a fact-checker. Write {MIN_CHALLENGES} to {MAX_CHALLENGES} open verification questions, "
        "one per factual claim in the proposed answer, then answer each question yourself."
    ),
    user="Question: {{ case.text }}\nProposed answer: {{ proposal.text }}",
)
_CROSS_EXAMINER: Final = JinjaPrompt(
    system=(
        "You are a cross-examiner. Decide whether an independently obtained answer agrees with the "
        "proposed answer on the point the verification question asks about."
    ),
    user=(
        "Proposed answer: {{ proposal.text }}\n"
        "Verification question: {{ challenge }}\n"
        "Independent answer: {{ testimony.answer }} ({{ testimony.explanation }})"
    ),
)
_JUDGE_SYSTEM: Final = (
    "You are the judge in a chain of verification. You are given a question, a proposed answer, and "
    "evidence from independent witnesses who never saw the proposed answer. Where the evidence "
    "contradicts the proposal, write a corrected answer from the evidence; where it supports the "
    "proposal, confirm it. Give the verdict, a confidence between 0 and 1, and the evidence that decided it."
)
_UNANSWERED: Final = "{% for item in evidence.unanswered %}\n- Unanswered: {{ item.challenge }}\n{% endfor %}"
_JUDGE: Final = JinjaPrompt(
    system=_JUDGE_SYSTEM,
    user=(
        "Question: {{ case.text }}\nProposed answer: {{ proposal.text }}\n\nEvidence:\n"
        "{% for item in evidence.answered %}\n"
        "- Q: {{ item.challenge }}\n"
        "  A: {{ item.testimony.answer }} ({{ item.testimony.explanation }})\n"
        "{% endfor %}" + _UNANSWERED
    ),
)
_CROSS_EXAMINED_JUDGE: Final = JinjaPrompt(
    system=_JUDGE_SYSTEM,
    user=(
        "Question: {{ case.text }}\nProposed answer: {{ proposal.text }}\n\nEvidence:\n"
        "{% for item in evidence.answered %}\n"
        "- Q: {{ item.challenge }}\n"
        "  A: {{ item.testimony.statement.answer }} ({{ item.testimony.statement.explanation }})\n"
        "  Cross-examination: {{ 'consistent' if item.testimony.finding.consistent else 'CONTRADICTS the proposal' }}"
        " - {{ item.testimony.finding.note }}\n"
        "{% endfor %}" + _UNANSWERED
    ),
)


@dataclass(frozen=True, slots=True)
class FactualQAPrompts:
    """Every prompt the recipe uses, typed by the brief its role receives.

    Override any of them with :func:`dataclasses.replace`, using a :class:`JinjaPrompt` or a
    typed ``(brief) -> Messages`` function.
    """

    proponent: Prompt[CaseBrief[Question]] = _PROPONENT
    contrarian: Prompt[ContrarianBrief[Question, Answer]] = _CONTRARIAN
    witness: Prompt[WitnessBrief[Question, str]] = _WITNESS
    panel: Prompt[PanelBrief[Question, str]] = _PANEL
    inquisitor: Prompt[ContrarianBrief[Question, Answer]] = _INQUISITOR
    cross_examiner: Prompt[CrossExaminationBrief[Question, Answer, str, WitnessAnswer]] = _CROSS_EXAMINER
    judge: Prompt[JudgeBrief[Question, Answer, str, WitnessAnswer]] = _JUDGE
    cross_examined_judge: Prompt[JudgeBrief[Question, Answer, str, CrossExamined[WitnessAnswer, Consistency]]] = (
        _CROSS_EXAMINED_JUDGE
    )


DEFAULT_PROMPTS: Final = FactualQAPrompts()


@dataclass(frozen=True, slots=True)
class FactualQARoles:
    """Which LLM plays which role. Each may be a different provider and model.

    The two-step panel is played by the witness's LLM, and the joint inquisitor by the contrarian's.
    """

    proponent: StructuredLLM
    contrarian: StructuredLLM
    witness: StructuredLLM
    judge: StructuredLLM


class FactualQAProcedure(StrEnum):
    """The CoVe execution variants this recipe can be built with (see :func:`build_factual_qa`)."""

    FACTORED = auto()
    TWO_STEP = auto()
    JOINT = auto()


type FactualQA = ChainOfVerification[Question, Answer, str, WitnessAnswer, VerifiedAnswer]
type CrossExaminedFactualQA = ChainOfVerification[
    Question, Answer, str, CrossExamined[WitnessAnswer, Consistency], VerifiedAnswer
]


def build_factual_qa(
    roles: FactualQARoles,
    *,
    procedure: FactualQAProcedure = FactualQAProcedure.FACTORED,
    prompts: FactualQAPrompts = DEFAULT_PROMPTS,
    limits: Limits = DEFAULT_LIMITS,
    observers: Sequence[Observer] = (),
) -> FactualQA:
    """Build the factual-QA chain of verification.

    Parameters
    ----------
    roles
        The LLM behind each role.
    procedure
        The CoVe execution variant. ``FACTORED`` (the default) keeps every witness independent.
    prompts
        The recipe's prompts; override any with :func:`dataclasses.replace`.
    limits
        The engine's bounds.
    observers
        Receive every event of every run.

    Returns
    -------
    FactualQA
        An engine to ``run`` or ``run_many`` questions through.

    """
    examination: Procedure[Question, Answer, str, WitnessAnswer]
    match procedure:
        case FactualQAProcedure.FACTORED:
            examination = Factored(
                contrarian=_contrarian(roles, prompts),
                witness=llm_step(llm=roles.witness, prompt=prompts.witness, response_model=WitnessAnswer),
            )
        case FactualQAProcedure.TWO_STEP:
            examination = TwoStep(
                contrarian=_contrarian(roles, prompts),
                panel=llm_step(
                    llm=roles.witness, prompt=prompts.panel, response_model=PanelAnswers, parse=_panel_answers
                ),
            )
        case FactualQAProcedure.JOINT:
            examination = Joint(
                inquisitor=llm_step(
                    llm=roles.contrarian, prompt=prompts.inquisitor, response_model=Inquiry, parse=_inquiry_record
                ),
            )
        case _:
            assert_never(procedure)
    return ChainOfVerification(
        proponent=llm_step(llm=roles.proponent, prompt=prompts.proponent, response_model=Answer),
        procedure=examination,
        judge=llm_step(llm=roles.judge, prompt=prompts.judge, response_model=VerifiedAnswer),
        limits=limits,
        observers=observers,
    )


def build_cross_examined_factual_qa(
    roles: FactualQARoles,
    *,
    cross_examiner: StructuredLLM,
    prompts: FactualQAPrompts = DEFAULT_PROMPTS,
    limits: Limits = DEFAULT_LIMITS,
    observers: Sequence[Observer] = (),
) -> CrossExaminedFactualQA:
    """Build the factor+revise variant, where a cross-examiner checks each testimony against the proposal.

    The paper's best variant on long-form answers. It costs one extra call per challenge, and its
    evidence carries each witness's statement together with the cross-examiner's finding.
    """
    return ChainOfVerification(
        proponent=llm_step(llm=roles.proponent, prompt=prompts.proponent, response_model=Answer),
        procedure=FactorRevise(
            contrarian=_contrarian(roles, prompts),
            witness=llm_step(llm=roles.witness, prompt=prompts.witness, response_model=WitnessAnswer),
            cross_examiner=llm_step(llm=cross_examiner, prompt=prompts.cross_examiner, response_model=Consistency),
        ),
        judge=llm_step(llm=roles.judge, prompt=prompts.cross_examined_judge, response_model=VerifiedAnswer),
        limits=limits,
        observers=observers,
    )


class FactualQAConfig(BaseModel):
    """The recipe as it appears in YAML or the environment: one LLM per role, the procedure, the limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proponent: RoleConfig
    contrarian: RoleConfig
    witness: RoleConfig
    judge: RoleConfig
    procedure: FactualQAProcedure = FactualQAProcedure.FACTORED
    limits: Annotated[Limits, Field(default_factory=Limits)]


@asynccontextmanager
async def open_factual_qa(
    config: FactualQAConfig,
    *,
    prompts: FactualQAPrompts = DEFAULT_PROMPTS,
    observers: Sequence[Observer] = (),
) -> AsyncGenerator[FactualQA]:
    """Open one agent per role, named after it, build the recipe, and close every agent on exit, even on error."""
    async with AsyncExitStack() as stack:
        roles = FactualQARoles(
            proponent=await stack.enter_async_context(open_role(config.proponent, name=Role.PROPONENT)),
            contrarian=await stack.enter_async_context(open_role(config.contrarian, name=Role.CONTRARIAN)),
            witness=await stack.enter_async_context(open_role(config.witness, name=Role.WITNESS)),
            judge=await stack.enter_async_context(open_role(config.judge, name=Role.JUDGE)),
        )
        yield build_factual_qa(
            roles, procedure=config.procedure, prompts=prompts, limits=config.limits, observers=observers
        )


def _contrarian(
    roles: FactualQARoles, prompts: FactualQAPrompts
) -> LLMStep[ContrarianBrief[Question, Answer], Challenges, list[str]]:
    return llm_step(llm=roles.contrarian, prompt=prompts.contrarian, response_model=Challenges, parse=_questions)


def _questions(challenges: Challenges) -> list[str]:
    return challenges.questions


def _panel_answers(panel: PanelAnswers) -> list[WitnessAnswer]:
    return panel.answers


def _inquiry_record(inquiry: Inquiry) -> list[Answered[str, WitnessAnswer]]:
    return [
        Answered(
            challenge=item.question,
            testimony=WitnessAnswer(answer=item.answer, explanation=item.explanation),
        )
        for item in inquiry.items
    ]
