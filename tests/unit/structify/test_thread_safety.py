from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import instructor
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr

from omniadapters.core.models import (
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from omniadapters.structify.adapters.openai import OpenAIAdapter
from omniadapters.structify.models import InstructorConfig


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
        errors: list[Exception] = []

        def access_client() -> None:
            try:
                client = adapter.client
                clients.append(client)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(100):
            thread = threading.Thread(target=access_client)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        assert len(clients) == 100
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
        errors: list[Exception] = []

        def access_instructor() -> None:
            try:
                instructor = adapter.instructor
                instructors.append(instructor)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(100):
            thread = threading.Thread(target=access_instructor)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        assert len(instructors) == 100
        assert all(instructor is instructors[0] for instructor in instructors), "Multiple instructor instances created!"

    def test_stress_test_concurrent_access(self) -> None:
        provider_config = OpenAIProviderConfig(api_key=SecretStr("test_key"))
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        results: dict[str, list[AsyncOpenAI | instructor.AsyncInstructor | Exception]] = {
            "clients": [],
            "instructors": [],
            "errors": [],
        }
        lock = threading.Lock()

        def access_both_properties() -> None:
            try:
                client = adapter.client
                instructor = adapter.instructor
                with lock:
                    results["clients"].append(client)
                    results["instructors"].append(instructor)
            except Exception as e:
                with lock:
                    results["errors"].append(e)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(access_both_properties) for _ in range(500)]

            for future in futures:
                future.result()

        assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"

        assert len(results["clients"]) == 500
        assert all(client is results["clients"][0] for client in results["clients"])

        assert len(results["instructors"]) == 500
        assert all(instructor is results["instructors"][0] for instructor in results["instructors"])

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
            instructor = adapter.instructor
            clients.append(client)
            instructors.append(instructor)

        tasks = [access_properties_async() for _ in range(50)]
        await asyncio.gather(*tasks)

        assert all(client is clients[0] for client in clients)
        assert all(instructor is instructors[0] for instructor in instructors)

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
