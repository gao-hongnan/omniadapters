from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from omniadapters.core.models import OpenAIProviderConfig
from omniadapters.pydantic_ai.adapter import PydanticAIAdapter


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []


class _AsyncCloseClient:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    async def close(self) -> None:
        self._rec.calls.append("close")


class _AsyncACloseClient:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    async def aclose(self) -> None:
        self._rec.calls.append("aclose")


class _GeminiAio:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    async def aclose(self) -> None:
        self._rec.calls.append("aio.aclose")


class _GeminiClient:
    def __init__(self, rec: _Recorder) -> None:
        self.aio = _GeminiAio(rec)


class _CtxClient:
    def __init__(self, rec: _Recorder) -> None:
        self._rec = rec

    async def __aenter__(self) -> "_CtxClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._rec.calls.append("__aexit__")


class _FakeModel:
    def __init__(self, client: object | None) -> None:
        self.client = client


def _make_adapter() -> PydanticAIAdapter:
    return PydanticAIAdapter(
        provider_config=OpenAIProviderConfig(api_key=SecretStr("sk-test")),
        model_name="gpt-4o-mini",
    )


@pytest.mark.unit
class TestPydanticAIAdapterAclose:
    @pytest.mark.asyncio
    async def test_noop_when_model_never_built(self) -> None:
        adapter = _make_adapter()
        # Must not raise; idempotent before any create_agent call.
        await adapter.aclose()
        assert adapter._model is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_factory,expected_call",
        [
            (_AsyncCloseClient, "close"),
            (_AsyncACloseClient, "aclose"),
            (_GeminiClient, "aio.aclose"),
            (_CtxClient, "__aexit__"),
        ],
        ids=["close", "aclose", "gemini-aio", "aexit"],
    )
    async def test_dispatches_to_each_protocol_shape(
        self,
        client_factory: type,
        expected_call: str,
    ) -> None:
        adapter = _make_adapter()
        rec = _Recorder()
        adapter._model = _FakeModel(client=client_factory(rec))  # type: ignore[assignment]

        await adapter.aclose()

        assert rec.calls == [expected_call]
        assert adapter._model is None

    @pytest.mark.asyncio
    async def test_handles_model_without_client_attribute(self) -> None:
        adapter = _make_adapter()

        class _NoClientModel: ...

        adapter._model = _NoClientModel()  # type: ignore[assignment]
        await adapter.aclose()  # must not raise
        assert adapter._model is None

    @pytest.mark.asyncio
    async def test_handles_client_attribute_is_none(self) -> None:
        adapter = _make_adapter()
        adapter._model = _FakeModel(client=None)  # type: ignore[assignment]
        await adapter.aclose()
        assert adapter._model is None

    @pytest.mark.asyncio
    async def test_idempotent(self) -> None:
        adapter = _make_adapter()
        rec = _Recorder()
        adapter._model = _FakeModel(client=_AsyncACloseClient(rec))  # type: ignore[assignment]

        await adapter.aclose()
        await adapter.aclose()  # second call is a no-op

        assert rec.calls == ["aclose"]
        assert adapter._model is None

    @pytest.mark.asyncio
    async def test_protocol_priority_gemini_before_aclose(self) -> None:
        """A client exposing both ``aio.aclose`` and ``aclose`` matches Gemini first."""
        adapter = _make_adapter()
        rec = _Recorder()

        class _Both:
            def __init__(self) -> None:
                self.aio = _GeminiAio(rec)

            async def aclose(self) -> None:
                rec.calls.append("aclose")

        adapter._model = _FakeModel(client=_Both())  # type: ignore[assignment]
        await adapter.aclose()

        assert rec.calls == ["aio.aclose"]
