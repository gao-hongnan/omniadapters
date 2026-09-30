"""Execution limits: every bound a chain-of-verification run enforces, as one validated config model."""

from __future__ import annotations

import math
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# `total * (1 - quorum)` can land a hair under an integer in binary floating point (5 * 0.2 -> 0.999...).
_ROUNDING_SLACK: Final = 1e-9


class Limits(BaseModel):
    """Every bound a run enforces.

    A frozen config model: nest it in your application's settings to load it from YAML or the
    environment. :data:`DEFAULT_LIMITS` is the single owner of the defaults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrent_steps: Annotated[
        int, Field(ge=1, le=256, description="Step calls in flight at once, across every run of one engine.")
    ] = 8
    max_concurrent_runs: Annotated[int, Field(ge=1, le=256, description="Cases `run_many` runs at once.")] = 4
    timeout_s: Annotated[
        float,
        Field(gt=0, le=900, description="Deadline for one proponent, contrarian, panel, inquisitor or judge call."),
    ] = 120.0
    witness_timeout_s: Annotated[
        float, Field(gt=0, le=900, description="Deadline for one witness or cross-examiner call.")
    ] = 60.0
    min_challenges: Annotated[
        int, Field(ge=0, le=100, description="Fewest challenges a hearing accepts; 0 allows a vacuous hearing.")
    ] = 1
    max_challenges: Annotated[
        int, Field(ge=1, le=100, description="Most challenges a hearing accepts: the cost guard on fan-out.")
    ] = 12
    quorum: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Share of challenges that must be answered; 1.0 ends the hearing on the first unanswered one.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def _challenge_window_is_ordered(self) -> Self:
        if self.min_challenges > self.max_challenges:
            msg = f"min_challenges ({self.min_challenges}) exceeds max_challenges ({self.max_challenges})"
            raise ValueError(msg)
        return self

    def tolerated_unanswered(self, total: int) -> int:
        """Return how many of ``total`` challenges may go unanswered before the quorum is lost."""
        return math.floor(total * (1.0 - self.quorum) + _ROUNDING_SLACK)


DEFAULT_LIMITS: Final = Limits()
