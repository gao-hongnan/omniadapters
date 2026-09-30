"""Closed vocabularies of the chain-of-verification courtroom."""

from __future__ import annotations

from enum import StrEnum, auto


class Role(StrEnum):
    """Who takes the stand in a chain-of-verification run.

    Each role is one step of Dhuliawala et al. (2023), arXiv:2309.11495:

    - ``PROPONENT`` drafts the baseline response, the *proposal*.
    - ``CONTRARIAN`` plans verification questions, the *challenges*.
    - ``WITNESS`` answers one challenge in isolation (factored execution).
    - ``PANEL`` answers every challenge in one call, without the proposal (two-step execution).
    - ``INQUISITOR`` challenges and answers in one call that sees the proposal (joint execution).
    - ``CROSS_EXAMINER`` checks one testimony against the proposal (factor+revise).
    - ``JUDGE`` writes the final verified response, the *ruling*.
    """

    PROPONENT = auto()
    CONTRARIAN = auto()
    WITNESS = auto()
    PANEL = auto()
    INQUISITOR = auto()
    CROSS_EXAMINER = auto()
    JUDGE = auto()


class Disposition(StrEnum):
    """How a run's hearings ended.

    - ``SINGLE_HEARING``: no retrial was configured, so one hearing produced the ruling.
    - ``SETTLED``: the retrial policy let a ruling stand within the hearing limit.
    - ``HEARING_LIMIT``: the retrial policy still wanted another hearing when the limit was reached;
      treat the ruling as unsettled (for example, route it to human review).
    """

    SINGLE_HEARING = auto()
    SETTLED = auto()
    HEARING_LIMIT = auto()


class UnansweredReason(StrEnum):
    """Why a challenge went unanswered: its step ran out of time, or raised."""

    TIMEOUT = auto()
    ERROR = auto()
