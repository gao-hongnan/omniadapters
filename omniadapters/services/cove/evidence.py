"""The evidence a hearing produces: one outcome per challenge, answered or not."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .enums import UnansweredReason


@dataclass(frozen=True, slots=True)
class Answered[ChallengeT, TestimonyT]:
    """A challenge and the testimony given for it."""

    challenge: ChallengeT
    testimony: TestimonyT
    kind: Literal["answered"] = field(default="answered", init=False)


@dataclass(frozen=True, slots=True)
class Unanswered[ChallengeT]:
    """A challenge whose witness timed out or raised. Kept in the evidence, never dropped."""

    challenge: ChallengeT
    reason: UnansweredReason
    error_type: str
    message: str
    kind: Literal["unanswered"] = field(default="unanswered", init=False)


type Outcome[ChallengeT, TestimonyT] = Answered[ChallengeT, TestimonyT] | Unanswered[ChallengeT]


@dataclass(frozen=True, slots=True)
class Evidence[ChallengeT, TestimonyT]:
    """Every challenge's outcome, in the order the contrarian raised them.

    ``answered`` and ``unanswered`` are views for templates and judges that treat the two
    kinds separately; ``outcomes`` keeps them interleaved in challenge order.
    """

    outcomes: tuple[Outcome[ChallengeT, TestimonyT], ...]

    @property
    def answered(self) -> tuple[Answered[ChallengeT, TestimonyT], ...]:
        """The challenges that were answered, in challenge order."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, Answered))

    @property
    def unanswered(self) -> tuple[Unanswered[ChallengeT], ...]:
        """The challenges that went unanswered, in challenge order."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, Unanswered))


@dataclass(frozen=True, slots=True)
class CrossExamined[StatementT, FindingT]:
    """Factor+revise testimony: a witness's independent statement and the cross-examiner's finding on it."""

    statement: StatementT
    finding: FindingT
