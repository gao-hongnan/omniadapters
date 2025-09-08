from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

from omniadapters.core.protocols import AsyncCloseable, AsyncContextManager
from omniadapters.core.types import ClientResponseT, ClientT, MessageParam, ProviderConfigT, StreamResponseT

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionClientParams


class BaseAdapter(ABC, Generic[ProviderConfigT, ClientT, ClientResponseT, StreamResponseT]):
    def __init__(
        self,
        *,
        provider_config: ProviderConfigT,
        completion_params: CompletionClientParams | None = None,
    ) -> None:
        self.provider_config = provider_config
        self.completion_params: dict[str, Any] = completion_params.model_dump() if completion_params else {}
        self._client: ClientT | None = None
        self._client_lock = threading.Lock()

    @property
    def client(self) -> ClientT:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> ClientT: ...

    @overload
    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ClientResponseT: ...

    @overload
    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> StreamResponseT: ...

    async def agenerate(
        self,
        messages: list[MessageParam],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ClientResponseT | StreamResponseT:
        merged_params = {**self.completion_params, **kwargs, "stream": stream}

        if stream:
            return cast(StreamResponseT, self._agenerate_stream(messages, **merged_params))
        else:
            return await self._agenerate(messages, **merged_params)

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

    async def aclose(self) -> None:
        if self._client is None:
            return

        if isinstance(self._client, AsyncCloseable):
            await self._client.close()
        elif isinstance(self._client, AsyncContextManager):
            await self._client.__aexit__(None, None, None)

        self._client = None
        self._instructor = None
