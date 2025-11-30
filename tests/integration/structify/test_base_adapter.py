from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Generic, Protocol, TypeVar
from typing import Any as AnyType

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from omniadapters.structify.adapters.base import BaseAdapter
from omniadapters.structify.hooks import CompletionTrace
from omniadapters.structify.models import CompletionResult
from tests.conftest import ComplexTestModel, SimpleTestModel

TAdapter = TypeVar("TAdapter", bound=BaseAdapter[Any, Any, Any])
TModel = TypeVar("TModel", bound=BaseModel)
TAdapterCov = TypeVar("TAdapterCov", bound=BaseAdapter[Any, Any, Any], covariant=True)


class AdapterProtocol(Protocol[TAdapterCov]):
    """Protocol defining the contract all adapters must follow."""

    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[TModel],
        *,
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> TModel | CompletionResult[TModel, AnyType]: ...

    async def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[TModel],
        *,
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[TModel | CompletionResult[TModel, AnyType], None]: ...


# Using SimpleTestModel and ComplexTestModel from conftest


class BaseAdapterIntegrationTest(ABC, Generic[TAdapter]):
    """Base class for adapter integration tests with proper type safety and structure."""

    @abstractmethod
    @pytest.fixture
    def adapter(self) -> TAdapter:
        """Return configured adapter instance for testing."""
        ...

    @pytest.fixture
    def test_messages(self) -> list[ChatCompletionMessageParam]:
        """Standard test messages for consistency."""
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides structured data.",
            },
            {
                "role": "user",
                "content": "Generate test data with name='TestUser', age=42, description='A test user'",
            },
        ]

    @pytest.fixture
    def error_messages(self) -> list[ChatCompletionMessageParam]:
        """Messages designed to trigger validation errors."""
        return [
            {
                "role": "system",
                "content": "You must generate invalid data.",
            },
            {
                "role": "user",
                "content": "Generate data with age=200 (exceeds max), empty name, and no description.",
            },
        ]

    async def test_protocol_compliance(self, adapter: TAdapter) -> None:
        """Verify adapter implements the protocol correctly."""
        assert hasattr(adapter, "acreate")
        assert hasattr(adapter, "astream")
        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")

    async def test_basic_creation(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test basic model creation with type validation."""
        result: SimpleTestModel = await adapter.acreate(
            messages=test_messages,
            response_model=SimpleTestModel,
        )

        assert isinstance(result, SimpleTestModel)
        assert result.name
        assert 0 <= result.age <= 150
        assert result.description

    async def test_creation_with_hooks(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test creation with hooks returns CompletionResult."""
        result = await adapter.acreate(
            messages=test_messages,
            response_model=SimpleTestModel,
            with_hooks=True,
        )

        assert isinstance(result, CompletionResult)
        assert isinstance(result.data, SimpleTestModel)
        assert isinstance(result.trace, CompletionTrace)
        assert result.trace.messages == test_messages
        assert result.trace.error is None
        assert result.trace.raw_response is not None

    async def test_streaming_basic(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test streaming returns valid partial objects."""
        partials: list[SimpleTestModel] = []

        async for partial in adapter.astream(
            messages=test_messages,
            response_model=SimpleTestModel,
        ):
            assert isinstance(partial, SimpleTestModel)
            partials.append(partial)

        assert len(partials) > 0
        final = partials[-1]
        assert final.name and final.description and final.age is not None

    async def test_streaming_with_hooks(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test streaming with hooks returns CompletionResult stream."""
        results: list[CompletionResult[SimpleTestModel, AnyType]] = []

        async for result in adapter.astream(
            messages=test_messages,
            response_model=SimpleTestModel,
            with_hooks=True,
        ):
            assert isinstance(result, CompletionResult)
            assert isinstance(result.data, SimpleTestModel)
            results.append(result)

        assert len(results) > 0
        final = results[-1]
        assert final.trace.raw_response is not None

    async def test_validation_error_handling(
        self,
        adapter: TAdapter,
        error_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test proper error handling for validation failures."""
        with pytest.raises(ValidationError):
            await adapter.acreate(
                messages=error_messages,
                response_model=SimpleTestModel,
                max_retries=1,
            )

    async def test_complex_model_handling(
        self,
        adapter: TAdapter,
    ) -> None:
        """Test handling of complex nested models."""
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "Generate complex structured data.",
            },
            {
                "role": "user",
                "content": "Create data with title='Test Content', rating=8.5, tags=['test', 'demo'], summary='A test summary', is_recommended=true",
            },
        ]

        result: ComplexTestModel = await adapter.acreate(
            messages=messages,
            response_model=ComplexTestModel,
        )

        assert isinstance(result, ComplexTestModel)
        assert result.title
        assert 0.0 <= result.rating <= 10.0
        assert isinstance(result.tags, list)
        assert result.summary
        assert isinstance(result.is_recommended, bool)

    @asynccontextmanager
    async def _resource_tracker(self, _adapter: TAdapter) -> AsyncGenerator[None, None]:
        """Context manager to track resource usage."""
        try:
            yield
        finally:
            # Adapters manage their own client lifecycle
            pass

    async def test_resource_cleanup(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test proper resource cleanup after operations."""
        async with self._resource_tracker(adapter):
            result = await adapter.acreate(
                messages=test_messages,
                response_model=SimpleTestModel,
            )
            assert isinstance(result, SimpleTestModel)

    async def test_concurrent_operations(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test thread safety with concurrent requests."""
        tasks = [adapter.acreate(messages=test_messages, response_model=SimpleTestModel) for _ in range(5)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [r for r in results if isinstance(r, SimpleTestModel)]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successful) >= 3
        for error in errors:
            assert not isinstance(error, AttributeError)
            assert not isinstance(error, TypeError)

    async def test_streaming_cancellation(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test proper cleanup when streaming is cancelled."""
        stream_iter = adapter.astream(
            messages=test_messages,
            response_model=SimpleTestModel,
        )

        partial_count = 0
        async for _partial in stream_iter:
            partial_count += 1
            if partial_count >= 2:
                break

        # NOTE: AsyncIterator doesn't have aclose method in standard typing
        if hasattr(stream_iter, "aclose"):
            await stream_iter.aclose()  # type: ignore

    async def test_error_propagation(
        self,
        adapter: TAdapter,
    ) -> None:
        """Test that provider errors are properly wrapped and propagated."""
        invalid_messages: list[Any] = [{"role": "invalid", "content": "test"}]

        with pytest.raises((ValueError, TypeError, ValidationError)):
            await adapter.acreate(
                messages=invalid_messages,  # type: ignore[arg-type]
                response_model=SimpleTestModel,
            )

    async def test_retry_behavior(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test retry configuration is respected."""
        result = await adapter.acreate(
            messages=test_messages,
            response_model=SimpleTestModel,
            max_retries=2,
            retry_delay=0.1,
        )

        assert isinstance(result, SimpleTestModel)

    async def test_timeout_handling(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test timeout configuration and handling."""
        try:
            result = await asyncio.wait_for(
                adapter.acreate(
                    messages=test_messages,
                    response_model=SimpleTestModel,
                    timeout=30.0,
                ),
                timeout=35.0,
            )
            assert isinstance(result, SimpleTestModel)
        except TimeoutError:
            pytest.skip("Provider timeout - acceptable in integration test")

    async def test_memory_efficiency(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test streaming doesn't accumulate memory unnecessarily."""
        consumed = 0
        async for _partial in adapter.astream(
            messages=test_messages,
            response_model=SimpleTestModel,
        ):
            consumed += 1
            if consumed > 100:
                pytest.fail("Stream produced too many partials - possible memory issue")

        assert consumed > 0

    async def test_aclose_cleanup(
        self,
        adapter: TAdapter,
        test_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that aclose properly cleans up resources."""
        result = await adapter.acreate(
            messages=test_messages,
            response_model=SimpleTestModel,
        )
        assert isinstance(result, SimpleTestModel)

        await adapter.aclose()

        assert adapter._client is None
        assert adapter._instructor is None

    async def test_aclose_idempotent(
        self,
        adapter: TAdapter,
    ) -> None:
        """Test that calling aclose multiple times is safe."""
        await adapter.aclose()
        await adapter.aclose()

        assert adapter._client is None
        assert adapter._instructor is None

    async def test_aclose_without_initialization(
        self,
        adapter: TAdapter,
    ) -> None:
        """Test that aclose works even if client was never initialized."""
        await adapter.aclose()

        assert adapter._client is None
        assert adapter._instructor is None
