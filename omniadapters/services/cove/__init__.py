"""Chain-of-Verification as a reusable engine, staged as a courtroom.

Chain-of-Verification (CoVe; Dhuliawala et al. 2023, arXiv:2309.11495) reduces hallucination in
four steps: draft a baseline response, plan verification questions, answer them independently,
and write a final verified response. This package makes the *proceedings* reusable. The engine
owns the order of play and who may see what. A recipe supplies the roles.

==================================  =================  ===========================  ==============
CoVe step                           Role               Sees (brief)                 Produces
==================================  =================  ===========================  ==============
1. baseline response                ``proponent``      :class:`CaseBrief`           the proposal
2. plan verifications               ``contrarian``     :class:`ContrarianBrief`     challenges
3. execute (factored)               ``witness``        :class:`WitnessBrief`        testimony
3. execute (two-step)               ``panel``          :class:`PanelBrief`          testimonies
2+3. plan and execute (joint)       ``inquisitor``     :class:`ContrarianBrief`     answered pairs
3b. cross-check (factor+revise)     ``cross_examiner`` :class:`CrossExaminationBrief` findings
4. final verified response          ``judge``          :class:`JudgeBrief`          the ruling
==================================  =================  ===========================  ==============

Witnesses and panels never see the proposal. That is the paper's key finding: verification
conditioned on the draft repeats its errors. It is enforced by their briefs' types, and by a
template check that runs when the recipe is built.

Quickstart
----------
A step is any ``async (brief) -> result``: an LLM call built with :func:`llm_step`, a database
lookup, a regex, a test runner::

    from omniadapters.services.cove import ChainOfVerification, Factored, JinjaPrompt, WitnessBrief, llm_step

    async def check_against_source(brief: WitnessBrief[Document, str]) -> Support:
        ...  # sees brief.case and brief.challenge; there is no brief.proposal to leak

    cove = ChainOfVerification[Document, Summary, str, Support, FaithfulSummary](
        proponent=llm_step(llm=writer, prompt=SUMMARISE, response_model=Summary),
        procedure=Factored(
            contrarian=llm_step(llm=skeptic, prompt=SPLIT_CLAIMS, response_model=Claims, parse=claim_list),
            witness=check_against_source,
        ),
        judge=llm_step(llm=editor, prompt=REWRITE, response_model=FaithfulSummary),
    )
    transcript = await cove.run(document)  # .ruling  .hearings  .disposition  .to_json()

Swap ``Factored`` for :class:`TwoStep`, :class:`Joint` or :class:`FactorRevise` to change the
paper variant, or implement :class:`Procedure` against the :class:`Court` to add your own.

Where to go next
----------------
- :mod:`~omniadapters.services.cove.backends`: :class:`~omniadapters.services.cove.backends.PydanticAIStructuredLLM`,
  a pydantic-ai agent behind the LLM port, and :class:`~omniadapters.services.cove.backends.RoleConfig`
  with ``open_role``, one agent per role from YAML or env, closed on exit.
- :mod:`~omniadapters.services.cove.testing`: ``ScriptedLLM``, for offline recipe tests.
- :mod:`~omniadapters.services.cove.events`: the event types observers receive.
- :mod:`~omniadapters.services.cove.recipes.factual_qa`: a complete recipe (the former
  ``playground/critic``).
"""

from __future__ import annotations

from .briefs import (
    CaseBrief,
    ContrarianBrief,
    CrossExaminationBrief,
    JudgeBrief,
    PanelBrief,
    Step,
    WitnessBrief,
)
from .court import Court, Stand
from .engine import ChainOfVerification, Reopen, Retrial
from .enums import Disposition, Role, UnansweredReason
from .errors import (
    BudgetExceededError,
    ChallengeCountError,
    CoVeError,
    CoVeRunError,
    PanelMismatchError,
    QuorumNotMetError,
    StepFailedError,
    StepTimeoutError,
    TemplateVariableError,
)
from .events import CoVeEvent, Observer, RunId, StepRecord, logging_observer
from .evidence import Answered, CrossExamined, Evidence, Outcome, Unanswered
from .limits import DEFAULT_LIMITS, Limits
from .llm import AsyncLimiter, Budgeted, CallBudget, LLMStep, StructuredLLM, Throttled, llm_step
from .procedures import Factored, FactorRevise, Joint, Procedure, TwoStep, blind
from .prompts import JinjaPrompt, Messages, Prompt
from .transcript import Hearing, Transcript

__all__ = [
    "DEFAULT_LIMITS",
    "Answered",
    "AsyncLimiter",
    "BudgetExceededError",
    "Budgeted",
    "CallBudget",
    "CaseBrief",
    "ChainOfVerification",
    "ChallengeCountError",
    "CoVeError",
    "CoVeEvent",
    "CoVeRunError",
    "ContrarianBrief",
    "Court",
    "CrossExaminationBrief",
    "CrossExamined",
    "Disposition",
    "Evidence",
    "FactorRevise",
    "Factored",
    "Hearing",
    "JinjaPrompt",
    "Joint",
    "JudgeBrief",
    "LLMStep",
    "Limits",
    "Messages",
    "Observer",
    "Outcome",
    "PanelBrief",
    "PanelMismatchError",
    "Procedure",
    "Prompt",
    "QuorumNotMetError",
    "Reopen",
    "Retrial",
    "Role",
    "RunId",
    "Stand",
    "Step",
    "StepFailedError",
    "StepRecord",
    "StepTimeoutError",
    "StructuredLLM",
    "TemplateVariableError",
    "Throttled",
    "Transcript",
    "TwoStep",
    "Unanswered",
    "UnansweredReason",
    "WitnessBrief",
    "blind",
    "llm_step",
    "logging_observer",
]
