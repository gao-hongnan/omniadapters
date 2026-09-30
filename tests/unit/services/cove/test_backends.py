"""Backends: role config from YAML/env, one structify client per role closed on exit, and pydantic-ai."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from omniadapters.services.cove.backends import PydanticAIStructuredLLM, RoleConfig, open_role
from omniadapters.structify.adapters.base import BaseAdapter

_OPENAI_ROLE: Final = {
    "provider_config": {"provider": "openai", "api_key": "sk-test-secret"},
    "completion_params": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
    "instructor_config": {"mode": "tool_call"},
}


def _gemini_role(provider_tag: str) -> dict[str, object]:
    return {
        "provider_config": {"provider": provider_tag, "api_key": "test"},
        "completion_params": {"provider": provider_tag, "model": "gemini-2.5-flash"},
        "instructor_config": {"mode": "genai_structured_outputs"},
    }


class _Verdict(BaseModel):
    answer: str
    confident: bool


@pytest.mark.unit
class TestRoleConfig:
    def test_parses_a_yaml_shaped_mapping_and_keeps_the_key_secret(self) -> None:
        config = RoleConfig.model_validate(_OPENAI_ROLE)

        assert config.provider_config.api_key.get_secret_value() == "sk-test-secret"
        assert "sk-test-secret" not in repr(config)

    def test_accepts_the_google_provider_tag(self) -> None:
        assert RoleConfig.model_validate(_gemini_role("google")).provider_config.provider == "google"

    def test_rejects_the_stale_gemini_provider_tag(self) -> None:
        with pytest.raises(ValidationError, match="'google'"):
            RoleConfig.model_validate(_gemini_role("gemini"))

    def test_rejects_a_provider_mismatch(self) -> None:
        mismatched = {**_OPENAI_ROLE, "completion_params": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}

        with pytest.raises(ValidationError, match="provider mismatch"):
            RoleConfig.model_validate(mismatched)

    def test_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError, match="prompts"):
            RoleConfig.model_validate({**_OPENAI_ROLE, "prompts": {"base_path": "prompts"}})


@pytest.mark.unit
class TestOpenRole:
    @pytest.mark.asyncio
    async def test_closes_the_client_on_exit(self) -> None:
        async with open_role(RoleConfig.model_validate(_OPENAI_ROLE)) as llm:
            assert isinstance(llm, BaseAdapter)
            assert llm.client is not None

        assert llm._client is None

    @pytest.mark.asyncio
    async def test_closes_the_client_on_error(self) -> None:
        opened: list[object] = []

        async def crash_while_role_is_open() -> None:
            async with open_role(RoleConfig.model_validate(_OPENAI_ROLE)) as llm:
                assert isinstance(llm, BaseAdapter)
                assert llm.client is not None
                opened.append(llm)
                msg = "judge crashed"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="judge crashed"):
            await crash_while_role_is_open()

        (adapter,) = opened
        assert isinstance(adapter, BaseAdapter)
        assert adapter._client is None


@pytest.mark.unit
class TestPydanticAIBackend:
    @pytest.mark.asyncio
    async def test_returns_the_response_model_with_system_messages_as_instructions(self) -> None:
        llm = PydanticAIStructuredLLM(Agent(TestModel()))

        with capture_run_messages() as messages:
            verdict = await llm.acreate(
                [
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "user", "content": "Capital of Australia?"},
                ],
                _Verdict,
            )

        assert isinstance(verdict, _Verdict)
        request = messages[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions == "Answer in one word."

    @pytest.mark.asyncio
    async def test_rejects_non_text_messages(self) -> None:
        llm = PydanticAIStructuredLLM(Agent(TestModel()))

        with pytest.raises(TypeError, match="text-only user messages"):
            await llm.acreate([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], _Verdict)
