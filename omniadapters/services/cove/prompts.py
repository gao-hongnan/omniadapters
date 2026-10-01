"""Prompts: how a brief becomes the messages a role is shown."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

import jinja2
import jinja2.meta
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Messages:
    """What a role is shown on one call: a system message and a user message.

    Two texts and nothing else, so every backend sends a prompt exactly as it was written. The
    pydantic-ai backend makes ``system`` the run's instructions and ``user`` its prompt.
    """

    system: str
    user: str


type Prompt[BriefT] = Callable[[BriefT], Messages]

# Prompts are plain text sent to an LLM, never HTML, so autoescaping would corrupt them.
_ENVIRONMENT: Final = jinja2.Environment(  # nosec B701
    undefined=jinja2.StrictUndefined,
    autoescape=False,  # noqa: S701 - prompts are plain text, never HTML
    trim_blocks=True,
    lstrip_blocks=True,
)


class JinjaPrompt:
    """A system template and a user template, rendered from a brief's fields.

    Templates see the brief's fields as top-level names: ``{{ case.text }}``, ``{{ proposal }}``,
    ``{{ challenge }}``, ``{% for item in evidence.answered %}``. An undefined name raises instead
    of rendering blank (``StrictUndefined``). The engine also checks :attr:`variables` against
    each role's brief when a recipe is built, so a witness prompt that references ``proposal``
    fails before the first LLM call.
    """

    __slots__ = ("_system", "_user", "_variables")

    def __init__(self, *, system: str, user: str) -> None:
        self._system = _ENVIRONMENT.from_string(system)
        self._user = _ENVIRONMENT.from_string(user)
        self._variables = _referenced(system) | _referenced(user)

    @classmethod
    def from_files(cls, directory: Path, *, system: str, user: str) -> Self:
        """Build a prompt from two template files in ``directory``."""
        return cls(
            system=(directory / system).read_text(encoding="utf-8"),
            user=(directory / user).read_text(encoding="utf-8"),
        )

    @property
    def variables(self) -> frozenset[str]:
        """Every top-level name the two templates reference."""
        return self._variables

    def __call__(self, brief: object) -> Messages:
        """Render both templates against ``brief``'s fields."""
        values = _fields_of(brief)
        return Messages(system=self._system.render(values), user=self._user.render(values))


def _referenced(source: str) -> frozenset[str]:
    return frozenset(jinja2.meta.find_undeclared_variables(_ENVIRONMENT.parse(source)))


def _fields_of(brief: object) -> dict[str, object]:
    if isinstance(brief, BaseModel):
        return {name: getattr(brief, name) for name in type(brief).model_fields}
    if dataclasses.is_dataclass(brief) and not isinstance(brief, type):
        return {item.name: getattr(brief, item.name) for item in dataclasses.fields(brief)}
    msg = f"a prompt brief must be a dataclass or pydantic model instance, got {type(brief).__name__}"
    raise TypeError(msg)
