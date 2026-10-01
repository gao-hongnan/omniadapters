"""Backends: a pydantic-ai agent behind the LLM port, and one agent per role from YAML or env, closed on exit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile

from omniadapters.services.cove import Messages
from omniadapters.services.cove.backends import OutputMode, PydanticAIStructuredLLM, RoleConfig, open_role

if TYPE_CHECKING:
    from collections.abc import Mapping

    from openai import AsyncOpenAI
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models.function import AgentInfo
    from pydantic_ai.output import StructuredOutputMode

_OPENAI_ROLE: Final = {
    "provider_config": {"provider": "openai", "api_key": "sk-test-secret"},
    "model_name": "gpt-4o-mini",
    "model_settings": {"temperature": 0.0, "max_tokens": 2000},
}
_CAPITAL: Final = Messages(system="Answer in one word.", user="Capital of Australia?")
_CANBERRA: Final = {"answer": "Canberra"}
_MISNAMED: Final = {"reply": "Canberra"}


class _Verdict(BaseModel):
    answer: str


def _answering(
    answer: Mapping[str, object], requests: list[AgentInfo], *, profile: ModelProfile | None = None
) -> FunctionModel:
    """Build a model that always gives ``answer``, in whatever form the run asks for, recording every request."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        requests.append(info)
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, dict(answer))])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    return FunctionModel(respond, profile=profile)


def _user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def _openai_client(llm: PydanticAIStructuredLLM) -> AsyncOpenAI:
    model = llm.agent.model
    assert isinstance(model, OpenAIChatModel)
    return model.client


@pytest.mark.unit
class TestPydanticAIStructuredLLM:
    @pytest.mark.asyncio
    async def test_sends_the_system_message_as_instructions_and_the_user_message_as_the_prompt(self) -> None:
        seen: list[tuple[str | None, list[str]]] = []

        async def capital(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((info.instructions, _user_prompts(messages)))
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, _CANBERRA)])

        verdict = await PydanticAIStructuredLLM(Agent(FunctionModel(capital))).acreate(_CAPITAL, _Verdict)

        assert verdict == _Verdict(answer="Canberra")
        assert seen == [("Answer in one word.", ["Capital of Australia?"])]

    @pytest.mark.asyncio
    async def test_an_empty_system_message_sends_no_instructions(self) -> None:
        requests: list[AgentInfo] = []
        llm = PydanticAIStructuredLLM(Agent(_answering(_CANBERRA, requests)))

        await llm.acreate(Messages(system="", user="Capital of Australia?"), _Verdict)

        assert [info.instructions for info in requests] == [None]

    @pytest.mark.asyncio
    async def test_adds_the_system_message_to_the_agents_own_instructions(self) -> None:
        requests: list[AgentInfo] = []
        agent = Agent(_answering(_CANBERRA, requests), instructions="Reply in English.")

        await PydanticAIStructuredLLM(agent).acreate(_CAPITAL, _Verdict)

        assert [info.instructions for info in requests] == ["Reply in English.\nAnswer in one word."]

    @pytest.mark.parametrize(
        ("output_mode", "model_default", "requested"),
        [
            pytest.param(OutputMode.AUTO, "native", "native", id="auto-follows-the-model"),
            pytest.param(OutputMode.TOOL, "native", "tool", id="tool"),
            pytest.param(OutputMode.NATIVE, "tool", "native", id="native"),
            pytest.param(OutputMode.PROMPTED, "tool", "prompted", id="prompted"),
        ],
    )
    @pytest.mark.asyncio
    async def test_asks_the_model_for_structured_output_in_the_chosen_mode(
        self, output_mode: OutputMode, model_default: StructuredOutputMode, requested: StructuredOutputMode
    ) -> None:
        requests: list[AgentInfo] = []
        profile = ModelProfile(default_structured_output_mode=model_default, supports_json_schema_output=True)
        llm = PydanticAIStructuredLLM(Agent(_answering(_CANBERRA, requests, profile=profile)), output_mode=output_mode)

        verdict = await llm.acreate(_CAPITAL, _Verdict)

        assert verdict == _Verdict(answer="Canberra")
        assert [info.model_request_parameters.output_mode for info in requests] == [requested]

    @pytest.mark.asyncio
    async def test_shows_the_model_why_its_response_failed_validation_and_takes_the_retry(self) -> None:
        retries: list[RetryPromptPart] = []

        async def misname_then_correct(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            retries.extend(part for part in messages[-1].parts if isinstance(part, RetryPromptPart))
            arguments = _CANBERRA if retries else _MISNAMED
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, arguments)])

        verdict = await PydanticAIStructuredLLM(Agent(FunctionModel(misname_then_correct))).acreate(_CAPITAL, _Verdict)

        assert verdict == _Verdict(answer="Canberra")
        (retry,) = retries
        assert not isinstance(retry.content, str)
        assert [(error["type"], error["loc"]) for error in retry.content] == [("missing", ("answer",))]

    @pytest.mark.asyncio
    async def test_raises_once_the_output_retries_are_spent(self) -> None:
        llm = PydanticAIStructuredLLM(Agent(_answering(_MISNAMED, [])))

        with pytest.raises(UnexpectedModelBehavior, match="output retries"):
            await llm.acreate(_CAPITAL, _Verdict)


@pytest.mark.unit
class TestRoleConfig:
    def test_parses_a_yaml_shaped_mapping_and_keeps_the_key_secret(self) -> None:
        config = RoleConfig.model_validate(_OPENAI_ROLE)

        assert config.provider_config.api_key.get_secret_value() == "sk-test-secret"
        assert "sk-test-secret" not in repr(config)
        assert config.model_settings == {"temperature": 0.0, "max_tokens": 2000}

    def test_defaults_to_the_models_own_output_mode_one_retry_and_no_settings(self) -> None:
        config = RoleConfig.model_validate(
            {"provider_config": _OPENAI_ROLE["provider_config"], "model_name": "o4-mini"}
        )

        assert (config.output_mode, config.output_retries, config.model_settings) == (OutputMode.AUTO, 1, {})

    def test_accepts_the_google_provider_tag(self) -> None:
        config = RoleConfig.model_validate(
            {"provider_config": {"provider": "google", "api_key": "test"}, "model_name": "gemini-2.5-flash"}
        )

        assert config.provider_config.provider == "google"

    def test_rejects_the_stale_gemini_provider_tag(self) -> None:
        with pytest.raises(ValidationError, match="'google'"):
            RoleConfig.model_validate(
                {"provider_config": {"provider": "gemini", "api_key": "test"}, "model_name": "gemini-2.5-flash"}
            )

    def test_rejects_the_instructor_era_shape(self) -> None:
        instructor_era = {
            "provider_config": _OPENAI_ROLE["provider_config"],
            "completion_params": {"provider": "openai", "model": "gpt-4o-mini"},
            "instructor_config": {"mode": "tool_call"},
        }

        with pytest.raises(ValidationError, match="instructor_config"):
            RoleConfig.model_validate(instructor_era)

    def test_rejects_a_misspelt_model_setting_instead_of_ignoring_it(self) -> None:
        with pytest.raises(ValidationError, match="temprature"):
            RoleConfig.model_validate({**_OPENAI_ROLE, "model_settings": {"temprature": 0.0}})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("model_name", "", id="empty-model-name"),
            pytest.param("output_mode", "json", id="unknown-output-mode"),
            pytest.param("output_retries", -1, id="negative-retries"),
            pytest.param("output_retries", 11, id="retries-past-the-cost-guard"),
        ],
    )
    def test_rejects_an_out_of_range_field(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError, match=field):
            RoleConfig.model_validate({**_OPENAI_ROLE, field: value})


@pytest.mark.unit
class TestOpenRole:
    @pytest.mark.asyncio
    async def test_names_the_agent_after_the_role(self) -> None:
        async with open_role(RoleConfig.model_validate(_OPENAI_ROLE), name="witness") as llm:
            assert llm.agent.name == "witness"

    @pytest.mark.asyncio
    async def test_runs_the_model_with_the_roles_settings_and_output_mode(self) -> None:
        config = RoleConfig.model_validate({**_OPENAI_ROLE, "output_mode": "native"})
        requests: list[AgentInfo] = []

        async with open_role(config) as llm:
            with llm.agent.override(model=_answering(_CANBERRA, requests)):
                await llm.acreate(_CAPITAL, _Verdict)

        assert [(info.model_settings, info.model_request_parameters.output_mode) for info in requests] == [
            ({"temperature": 0.0, "max_tokens": 2000}, "native")
        ]

    @pytest.mark.parametrize("output_retries", [0, 3])
    @pytest.mark.asyncio
    async def test_gives_up_after_the_roles_output_retries(self, output_retries: int) -> None:
        config = RoleConfig.model_validate({**_OPENAI_ROLE, "output_retries": output_retries})
        requests: list[AgentInfo] = []

        async with open_role(config) as llm:
            with llm.agent.override(model=_answering(_MISNAMED, requests)), pytest.raises(UnexpectedModelBehavior):
                await llm.acreate(_CAPITAL, _Verdict)

        assert len(requests) == 1 + config.output_retries

    @pytest.mark.asyncio
    async def test_closes_the_http_client_on_exit(self) -> None:
        async with open_role(RoleConfig.model_validate(_OPENAI_ROLE)) as llm:
            client = _openai_client(llm)
            assert not client.is_closed()

        assert client.is_closed()

    @pytest.mark.asyncio
    async def test_closes_the_http_client_on_error(self) -> None:
        clients: list[AsyncOpenAI] = []

        async def crash_while_role_is_open() -> None:
            async with open_role(RoleConfig.model_validate(_OPENAI_ROLE)) as llm:
                clients.append(_openai_client(llm))
                msg = "judge crashed"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="judge crashed"):
            await crash_while_role_is_open()

        (client,) = clients
        assert client.is_closed()
