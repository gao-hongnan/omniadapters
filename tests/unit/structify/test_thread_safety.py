from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import instructor
import pytest
from pydantic import SecretStr

from omniadapters.core.models import (
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.structify.adapters.openai import OpenAIAdapter
from omniadapters.structify.models import InstructorConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_THREAD_COUNT = 100
_STRESS_THREAD_COUNT = 500


@pytest.mark.unit
class TestAdapterThreadSafety:
    def test_concurrent_client_access_thread_safety(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        clients: list[AsyncOpenAI] = []
        lock = threading.Lock()
        unhandled_errors: list[BaseException] = []

        def record_unhandled(args: threading.ExceptHookArgs) -> None:
            if args.exc_value is not None:
                with lock:
                    unhandled_errors.append(args.exc_value)

        def access_client() -> None:
            client = adapter.client
            with lock:
                clients.append(client)

        old_hook = threading.excepthook
        threading.excepthook = record_unhandled
        try:
            threads = [threading.Thread(target=access_client) for _ in range(_THREAD_COUNT)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            threading.excepthook = old_hook

        assert len(unhandled_errors) == 0, f"Errors occurred: {unhandled_errors}"
        assert len(clients) == _THREAD_COUNT
        assert all(client is clients[0] for client in clients), "Multiple client instances created!"

    def test_concurrent_instructor_access_thread_safety(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        instructors: list[instructor.AsyncInstructor] = []
        lock = threading.Lock()
        unhandled_errors: list[BaseException] = []

        def record_unhandled(args: threading.ExceptHookArgs) -> None:
            if args.exc_value is not None:
                with lock:
                    unhandled_errors.append(args.exc_value)

        def access_instructor() -> None:
            instr = adapter.instructor
            with lock:
                instructors.append(instr)

        old_hook = threading.excepthook
        threading.excepthook = record_unhandled
        try:
            threads = [threading.Thread(target=access_instructor) for _ in range(_THREAD_COUNT)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            threading.excepthook = old_hook

        assert len(unhandled_errors) == 0, f"Errors occurred: {unhandled_errors}"
        assert len(instructors) == _THREAD_COUNT
        assert all(instr is instructors[0] for instr in instructors), "Multiple instructor instances created!"

    def test_stress_test_concurrent_access(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        clients: list[AsyncOpenAI | instructor.AsyncInstructor] = []
        instructors: list[AsyncOpenAI | instructor.AsyncInstructor] = []
        lock = threading.Lock()

        def access_both_properties() -> None:
            client = adapter.client
            instr = adapter.instructor
            with lock:
                clients.append(client)
                instructors.append(instr)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(access_both_properties) for _ in range(_STRESS_THREAD_COUNT)]
            for future in futures:
                future.result()

        assert len(clients) == _STRESS_THREAD_COUNT
        assert all(client is clients[0] for client in clients)

        assert len(instructors) == _STRESS_THREAD_COUNT
        assert all(instr is instructors[0] for instr in instructors)

    def test_initialization_happens_only_once(self) -> None:
        create_client_count = 0
        with_instructor_count = 0

        class TestAdapter(OpenAIAdapter):
            def _create_client(self) -> AsyncOpenAI:
                nonlocal create_client_count
                create_client_count += 1
                time.sleep(0.01)
                return super()._create_client()

            def _with_instructor(self) -> instructor.AsyncInstructor:
                nonlocal with_instructor_count
                with_instructor_count += 1
                time.sleep(0.01)
                return super()._with_instructor()

        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = TestAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        def access_properties() -> None:
            _ = adapter.client
            _ = adapter.instructor

        threads = [threading.Thread(target=access_properties) for _ in range(20)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert create_client_count == 1, f"_create_client called {create_client_count} times"
        assert with_instructor_count == 1, f"_with_instructor called {with_instructor_count} times"

    @pytest.mark.asyncio
    async def test_thread_safety_in_async_context(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        clients = []
        instructors = []

        async def access_properties_async() -> None:
            client = adapter.client
            instr = adapter.instructor
            clients.append(client)
            instructors.append(instr)

        tasks = [access_properties_async() for _ in range(50)]
        await asyncio.gather(*tasks)

        assert all(client is clients[0] for client in clients)
        assert all(instr is instructors[0] for instr in instructors)

    def test_no_deadlock_with_nested_access(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        class NestedAdapter(OpenAIAdapter):
            def _with_instructor(self) -> instructor.AsyncInstructor:
                _ = self.client
                return super()._with_instructor()

        adapter = NestedAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        test_instructor = adapter.instructor
        test_client = adapter.client

        assert test_instructor is not None
        assert test_client is not None
