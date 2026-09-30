"""Offline test doubles: run and assert on a recipe without a network or an API key."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai.types.chat import ChatCompletionMessageParam

    from .prompts import Messages


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One call a :class:`ScriptedLLM` received."""

    messages: tuple[ChatCompletionMessageParam, ...]
    response_model: type[BaseModel]

    @property
    def text(self) -> str:
        """Every text message's content, joined by newlines: what the role was shown."""
        return "\n".join(content for message in self.messages if isinstance(content := message.get("content"), str))


type Script = Callable[[RecordedCall], BaseModel]


class ScriptedLLM:
    """A :class:`~omniadapters.services.cove.llm.StructuredLLM` that answers every call with ``script``.

    Each call is recorded first, so tests can assert on exactly what every role was shown.

    Examples
    --------
    >>> llm = ScriptedLLM(lambda call: Answer(text="Canberra"))  # doctest: +SKIP
    >>> [call.text for call in llm.calls]  # doctest: +SKIP

    """

    __slots__ = ("_calls", "_script")

    def __init__(self, script: Script) -> None:
        self._script = script
        self._calls: list[RecordedCall] = []

    @property
    def calls(self) -> tuple[RecordedCall, ...]:
        """Every call received so far, in order."""
        return tuple(self._calls)

    async def acreate[ResponseT: BaseModel](self, messages: Messages, response_model: type[ResponseT]) -> ResponseT:
        """Record the call, then answer it with the script.

        Raises
        ------
        TypeError
            If the script answers with an instance of the wrong response model.

        """
        call = RecordedCall(messages=tuple(messages), response_model=response_model)
        self._calls.append(call)
        reply = self._script(call)
        if not isinstance(reply, response_model):
            msg = f"the script answered a {response_model.__name__} request with a {type(reply).__name__}"
            raise TypeError(msg)
        return reply
