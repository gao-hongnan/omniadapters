from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Generic

from instructor import Mode, handle_response_model

from omniadapters.core.protocols import AsyncCloseable, AsyncContextManager
from omniadapters.core.types import ClientResponseT, ClientT, MessageParam, ProviderConfigT, StreamResponseT

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionClientParams, CompletionResponse, StreamChunk


class BaseAdapter(ABC, Generic[ProviderConfigT, ClientT, ClientResponseT, StreamResponseT]):
    def __init__(self, *, provider_config: ProviderConfigT, completion_params: CompletionClientParams) -> None:
        self.provider_config = provider_config
        self.completion_params = completion_params
        self._client: ClientT | None = None
        self._client_lock = threading.Lock()

    @property
    def client(self) -> ClientT:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    @property
    @abstractmethod
    def instructor_mode(self) -> Mode: ...

    @abstractmethod
    def _create_client(self) -> ClientT: ...

    def _prepare_request(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> dict[str, Any]:
        _, formatted_params = handle_response_model(
            response_model=None,
            mode=self.instructor_mode,
            messages=messages,
            **kwargs,
        )
        return dict(formatted_params)

    async def agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> CompletionResponse[ClientResponseT]:
        merged_params = {**self.completion_params.model_dump(), **kwargs}
        raw_response = await self._agenerate(messages, **merged_params)
        return self._to_unified_response(raw_response)

    async def agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        merged_params = {**self.completion_params.model_dump(), **kwargs}
        stream = self._agenerate_stream(messages, **merged_params)
        async for raw_chunk in stream:
            if unified_chunk := self._to_unified_chunk(raw_chunk):
                yield unified_chunk

    @abstractmethod
    async def _agenerate(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> ClientResponseT: ...

    @abstractmethod
    def _agenerate_stream(
        self,
        messages: list[MessageParam],
        **kwargs: Any,
    ) -> StreamResponseT: ...

    @abstractmethod
    def _to_unified_response(self, response: ClientResponseT) -> CompletionResponse[ClientResponseT]: ...

    @abstractmethod
    def _to_unified_chunk(self, chunk: Any) -> StreamChunk | None: ...

    async def aclose(self) -> None:
        if self._client is None:
            return

        if isinstance(self._client, AsyncCloseable):
            await self._client.close()
        elif isinstance(self._client, AsyncContextManager):
            await self._client.__aexit__(None, None, None)

        self._client = None
