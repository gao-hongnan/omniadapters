from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Generic, Literal, cast, overload

from instructor import Mode, handle_response_model

from omniadapters.core.protocols import AsyncCloseable, AsyncContextManager
from omniadapters.core.types import (
    ClientMessageT,
    ClientResponseT,
    ClientT,
    MessageParam,
    ProviderConfigT,
    StreamChunkT,
)

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionClientParams, CompletionResponse, StreamChunk


class BaseAdapter(ABC, Generic[ProviderConfigT, ClientT, ClientMessageT, ClientResponseT, StreamChunkT]):
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

    def _format_messages(self, messages: list[MessageParam], **kwargs: Any) -> list[ClientMessageT]:
        _, formatted_params = handle_response_model(
            response_model=None,
            mode=self.instructor_mode,
            messages=messages,
            **kwargs,
        )

        # NOTE: GEMINI uses "contents"
        if "contents" in formatted_params:
            formatted_messages = cast(list[ClientMessageT], formatted_params.pop("contents"))
        else:
            # NOTE:OpenAI, Anthropic, Azure use "messages"
            formatted_messages = cast(list[ClientMessageT], formatted_params.pop("messages"))

        return formatted_messages

    @overload
    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> CompletionResponse[ClientResponseT]: ...

    @overload
    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...

    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse[ClientResponseT] | AsyncIterator[StreamChunk]:
        merged_params = {**self.completion_params.model_dump(), **kwargs}

        if stream:
            stream_response = await self._agenerate(messages, stream=True, **merged_params)
            return self._stream_generator(stream_response)
        else:
            raw_response = await self._agenerate(messages, stream=False, **merged_params)
            return self._to_unified_response(raw_response)

    async def _stream_generator(self, stream: AsyncIterator[StreamChunkT]) -> AsyncIterator[StreamChunk]:
        async for raw_chunk in stream:
            if unified_chunk := self._to_unified_chunk(raw_chunk):
                yield unified_chunk

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ClientResponseT: ...

    @overload
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunkT]: ...

    @abstractmethod
    async def _agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ClientResponseT | AsyncIterator[StreamChunkT]: ...

    @abstractmethod
    def _to_unified_response(self, response: ClientResponseT) -> CompletionResponse[ClientResponseT]: ...

    @abstractmethod
    def _to_unified_chunk(self, chunk: StreamChunkT) -> StreamChunk | None: ...

    async def aclose(self) -> None:
        if self._client is None:
            return

        if isinstance(self._client, AsyncCloseable):
            await self._client.close()
        elif isinstance(self._client, AsyncContextManager):
            await self._client.__aexit__(None, None, None)

        self._client = None
