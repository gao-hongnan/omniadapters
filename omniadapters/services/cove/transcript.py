"""What a run returns: every hearing, how the run ended, and the full step record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pydantic_core

if TYPE_CHECKING:
    from .enums import Disposition
    from .events import RunId, StepRecord
    from .evidence import Evidence


@dataclass(frozen=True, slots=True)
class Hearing[ProposalT, ChallengeT, TestimonyT, RulingT]:
    """One examination of a proposal and the ruling it produced."""

    number: int
    proposal: ProposalT
    evidence: Evidence[ChallengeT, TestimonyT]
    ruling: RulingT


@dataclass(frozen=True, slots=True)
class Transcript[CaseT, ProposalT, ChallengeT, TestimonyT, RulingT]:
    """The complete, JSON-serialisable record of one run.

    ``hearings`` is never empty. :attr:`ruling` is the last hearing's ruling, and ``disposition``
    says whether that ruling settled the case. ``record`` lists every step call with its timing
    and outcome, including failed and cancelled ones.
    """

    run_id: RunId
    case: CaseT
    hearings: tuple[Hearing[ProposalT, ChallengeT, TestimonyT, RulingT], ...]
    disposition: Disposition
    record: tuple[StepRecord, ...]

    @property
    def ruling(self) -> RulingT:
        """The final ruling: the last hearing's."""
        return self.hearings[-1].ruling

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the whole transcript.

        The recipe's own types must be pydantic models, dataclasses, or JSON primitives.
        """
        return pydantic_core.to_json(self, indent=indent).decode()
