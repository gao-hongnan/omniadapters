from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from omniadapters.core.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.structify import create_adapter

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from rich.console import Console

    from omniadapters.structify.adapters.anthropic import AnthropicAdapter
    from omniadapters.structify.adapters.gemini import GeminiAdapter
    from omniadapters.structify.adapters.openai import OpenAIAdapter

from ..config.settings import (
    CoVeVerifierConfig,
    PromptsConfig,
    ProviderAgnosticAgent,
)
from ..prompts import JinjaPromptRenderer, get_prompt_renderer
from .base import MaybeOrchestrator, UserQuery
from .components import (
    DraftResponse,
    FactCheckAnswer,
    JudgeVerdict,
    SkepticQuestions,
)


class CoVeCandidate(BaseModel):
    chain_of_thought: list[str]
    is_aligned: bool
    confidence: float = Field(..., ge=0.0, le=1.0)

    draft_answer: str = Field(...)
    verification_questions: list[str] = Field(...)
    verification_answers: list[str] = Field(...)
    revision_made: bool = Field(default=False)
    verdict: str = Field(...)

    @computed_field
    def reasoning(self) -> str:
        return "\n".join(self.chain_of_thought) if self.chain_of_thought else ""


class CoVeOrchestrator(MaybeOrchestrator[CoVeVerifierConfig, CoVeCandidate]):
    config: CoVeVerifierConfig
    _prompt_renderer: JinjaPromptRenderer = PrivateAttr()
    _console: Console | None = PrivateAttr(default=None)
    _question_label: str = PrivateAttr(default="Question")

    def __init__(
        self,
        config: CoVeVerifierConfig,
        console: Console | None = None,
        question_label: str = "Question",
    ) -> None:
        super().__init__(config=config)
        self._console = console
        self._question_label = question_label
        self._prompt_renderer = get_prompt_renderer(self.config.drafter.prompts.base_path)

    def _log(self, message: str) -> None:
        if self._console is not None:
            self._console.log(f"[bold cyan]{self._question_label}[/] {message}")

    def _build_messages(
        self, prompts_config: PromptsConfig, context_variables: dict[str, Any]
    ) -> list[ChatCompletionMessageParam]:
        system_prompt = self._prompt_renderer.render(
            template_path=prompts_config.system_prompt_path,
            variables=prompts_config.system_context_variables,
        )

        user_variables = {
            **prompts_config.user_context_variables,
            **context_variables,
        }
        user_prompt = self._prompt_renderer.render(
            template_path=prompts_config.user_prompt_path,
            variables=user_variables,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _create_adapter(self, agent_config: ProviderAgnosticAgent) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
        match (agent_config.provider_config, agent_config.completion_params):
            case (OpenAIProviderConfig() as pc, OpenAICompletionClientParams() as cp):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case (
                AnthropicProviderConfig() as pc,
                AnthropicCompletionClientParams() as cp,
            ):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case (GeminiProviderConfig() as pc, GeminiCompletionClientParams() as cp):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case _:
                msg = f"Unsupported provider combination: {agent_config.provider_config.provider}"
                raise ValueError(msg)

    async def _run_drafter_phase(self, user_query: UserQuery) -> DraftResponse:
        start = time.perf_counter()
        self._log(f"drafter start provider={self.config.drafter.provider_config.provider}")
        messages = self._build_messages(self.config.drafter.prompts, user_query.model_dump())
        adapter = self._create_adapter(self.config.drafter)
        result = await adapter.acreate(messages, DraftResponse)
        self._log(f"drafter done aligned={result.is_aligned} elapsed={time.perf_counter() - start:.2f}s")
        return result

    async def _run_skeptic_phase(self, draft: DraftResponse, user_query: UserQuery) -> SkepticQuestions:
        start = time.perf_counter()
        self._log(f"skeptic start provider={self.config.skeptic.provider_config.provider}")
        context_variables = {
            "draft_reasoning": draft.reasoning,
            "draft_is_aligned": draft.is_aligned,
            **user_query.model_dump(),
        }
        messages = self._build_messages(self.config.skeptic.prompts, context_variables)
        adapter = self._create_adapter(self.config.skeptic)
        result = await adapter.acreate(messages, SkepticQuestions)
        self._log(f"skeptic done questions={len(result.questions)} elapsed={time.perf_counter() - start:.2f}s")
        return result

    async def _run_fact_checker_phase(
        self,
        questions: list[str],
        user_query: UserQuery,
    ) -> list[FactCheckAnswer]:
        start = time.perf_counter()
        self._log(
            f"fact-check start provider={self.config.fact_checker.provider_config.provider} questions={len(questions)}"
        )

        async def _fact_check_question(index: int, question: str) -> FactCheckAnswer:
            question_start = time.perf_counter()
            self._log(f"fact-check {index}/{len(questions)} start")
            context_variables = {
                "verification_question": question,
                **user_query.model_dump(),
            }
            messages = self._build_messages(self.config.fact_checker.prompts, context_variables)
            adapter = self._create_adapter(self.config.fact_checker)
            result = await adapter.acreate(messages, FactCheckAnswer)
            self._log(
                f"fact-check {index}/{len(questions)} done "
                f"answer={result.answer} elapsed={time.perf_counter() - question_start:.2f}s"
            )
            return result

        tasks = [
            asyncio.create_task(_fact_check_question(index, question))
            for index, question in enumerate(questions, start=1)
        ]
        results = await asyncio.gather(*tasks)
        self._log(f"fact-check done elapsed={time.perf_counter() - start:.2f}s")
        return results

    async def _run_judge_phase(
        self,
        draft: DraftResponse,
        questions: list[str],
        fact_checks: list[FactCheckAnswer],
        user_query: UserQuery,
    ) -> JudgeVerdict:
        start = time.perf_counter()
        self._log(f"judge start provider={self.config.judge.provider_config.provider}")
        qa_pairs = list(zip(questions, fact_checks, strict=False))
        context_variables = {
            "draft_reasoning": draft.reasoning,
            "draft_is_aligned": draft.is_aligned,
            "qa_pairs": qa_pairs,
            **user_query.model_dump(),
        }
        messages = self._build_messages(self.config.judge.prompts, context_variables)
        adapter = self._create_adapter(self.config.judge)
        result = await adapter.acreate(messages, JudgeVerdict)
        self._log(
            f"judge done aligned={result.is_aligned} "
            f"confidence={result.confidence:.2f} elapsed={time.perf_counter() - start:.2f}s"
        )
        return result

    async def aexecute(self, user_query: UserQuery) -> CoVeCandidate:
        start = time.perf_counter()
        self._log("started")
        draft = await self._run_drafter_phase(user_query)
        questions_obj = await self._run_skeptic_phase(draft, user_query)
        fact_check_answers = await self._run_fact_checker_phase(questions_obj.questions, user_query)
        judge_verdict = await self._run_judge_phase(draft, questions_obj.questions, fact_check_answers, user_query)
        self._log(f"finished elapsed={time.perf_counter() - start:.2f}s")

        chain_of_thought = [
            f"[Drafter] {draft.reasoning}",
            f"[Skeptic] Generated {len(questions_obj.questions)} verification questions",
            *[
                f"[FactCheck] {q} -> {a.answer}: {a.brief_explanation}"
                for q, a in zip(questions_obj.questions, fact_check_answers, strict=False)
            ],
            f"[Judge] {judge_verdict.reasoning}",
        ]

        return CoVeCandidate(
            chain_of_thought=chain_of_thought,
            is_aligned=judge_verdict.is_aligned,
            confidence=judge_verdict.confidence,
            draft_answer=draft.reasoning,
            verification_questions=questions_obj.questions,
            verification_answers=[f"{fc.answer}: {fc.brief_explanation}" for fc in fact_check_answers],
            revision_made=judge_verdict.revision_made,
            verdict=(
                f"{'Revised' if judge_verdict.revision_made else 'Confirmed'}: "
                f"{'Aligned' if judge_verdict.is_aligned else 'Not aligned'}"
            ),
        )
