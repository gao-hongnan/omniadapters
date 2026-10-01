"""A recipe whose witness is plain code, not an LLM: invoice extraction checked against its own source.

This is the pattern for any verification that is a lookup, a regex, a database query or a test
run. The witness is an ``async`` function typed by :class:`WitnessBrief`, so it gets the engine's
bounds, quorum and audit record without a prompt. It also never sees the proposal, so an
extraction cannot vouch for itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Final, NewType, assert_never

import pytest
from pydantic import BaseModel, ConfigDict

from omniadapters.services.cove import (
    ChainOfVerification,
    Disposition,
    Factored,
    Limits,
    Messages,
    Reopen,
    Retrial,
    UnansweredReason,
    llm_step,
)
from omniadapters.services.cove.testing import ScriptedLLM
from tests.unit.services.cove._toy import wait_forever

if TYPE_CHECKING:
    from omniadapters.services.cove import (
        CaseBrief,
        ContrarianBrief,
        JudgeBrief,
        Step,
        StructuredLLM,
        WitnessBrief,
    )
    from omniadapters.services.cove.testing import RecordedCall

InvoiceText = NewType("InvoiceText", str)

_SOURCE: Final = InvoiceText("ACME Industrial Supplies Pty Ltd\nInvoice 0042\nTotal due: AUD 1,234.50\n")
_AMOUNT: Final = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


class Invoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor: str
    total: Decimal


class TotalCorrection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: Decimal


class InvoiceField(StrEnum):
    VENDOR = auto()
    TOTAL = auto()


@dataclass(frozen=True, slots=True)
class FieldClaim:
    field: InvoiceField
    value: str


@dataclass(frozen=True, slots=True)
class CheckedInvoice:
    invoice: Invoice
    re_extracted: tuple[InvoiceField, ...]
    needs_review: tuple[InvoiceField, ...]


def extraction_prompt(brief: CaseBrief[InvoiceText]) -> Messages:
    return Messages(system="Extract the vendor and the total.", user=brief.case)


def total_prompt(source: InvoiceText) -> Messages:
    return Messages(system="Copy the total due exactly as printed.", user=source)


async def claim_every_field(brief: ContrarianBrief[InvoiceText, Invoice]) -> tuple[FieldClaim, ...]:
    return (
        FieldClaim(InvoiceField.VENDOR, brief.proposal.vendor),
        FieldClaim(InvoiceField.TOTAL, str(brief.proposal.total)),
    )


async def grounded_in_source(brief: WitnessBrief[InvoiceText, FieldClaim]) -> bool:
    claim = brief.challenge
    match claim.field:
        case InvoiceField.VENDOR:
            return claim.value.casefold() in brief.case.casefold()
        case InvoiceField.TOTAL:
            printed = {Decimal(token.replace(",", "")) for token in _AMOUNT.findall(brief.case)}
            return Decimal(claim.value) in printed
        case _:
            assert_never(claim.field)


@dataclass(frozen=True, slots=True)
class InvoiceJudge:
    """Keep grounded fields, re-extract an ungrounded total, and flag what could not be checked."""

    reextract_total: Step[InvoiceText, TotalCorrection]

    async def __call__(self, brief: JudgeBrief[InvoiceText, Invoice, FieldClaim, bool]) -> CheckedInvoice:
        ungrounded = tuple(item.challenge.field for item in brief.evidence.answered if not item.testimony)
        unchecked = tuple(item.challenge.field for item in brief.evidence.unanswered)
        invoice = brief.proposal
        if InvoiceField.TOTAL in ungrounded:
            correction = await self.reextract_total(brief.case)
            invoice = invoice.model_copy(update={"total": correction.total})
        needs_review = unchecked + tuple(field for field in ungrounded if field is not InvoiceField.TOTAL)
        return CheckedInvoice(invoice=invoice, re_extracted=ungrounded, needs_review=needs_review)


def reconsider(ruling: CheckedInvoice) -> Reopen[Invoice] | None:
    """Re-verify an invoice only when this hearing changed it."""
    return Reopen(proposal=ruling.invoice) if ruling.re_extracted else None


def build_invoice_cove(
    extractor: StructuredLLM,
    *,
    witness: Step[WitnessBrief[InvoiceText, FieldClaim], bool] = grounded_in_source,
    limits: Limits | None = None,
) -> ChainOfVerification[InvoiceText, Invoice, FieldClaim, bool, CheckedInvoice]:
    return ChainOfVerification(
        proponent=llm_step(llm=extractor, prompt=extraction_prompt, response_model=Invoice),
        procedure=Factored(contrarian=claim_every_field, witness=witness),
        judge=InvoiceJudge(
            reextract_total=llm_step(llm=extractor, prompt=total_prompt, response_model=TotalCorrection)
        ),
        retrial=Retrial(max_hearings=3, reconsider=reconsider),
        limits=limits or Limits(),
    )


def _misread_extractor(call: RecordedCall) -> BaseModel:
    if call.response_model is Invoice:
        return Invoice(vendor="ACME Industrial Supplies Pty Ltd", total=Decimal("1243.50"))
    if call.response_model is TotalCorrection:
        return TotalCorrection(total=Decimal("1234.50"))
    raise AssertionError(call.response_model.__name__)


@pytest.mark.unit
class TestDeterministicWitness:
    @pytest.mark.asyncio
    async def test_a_misread_total_is_re_extracted_and_re_verified(self) -> None:
        extractor = ScriptedLLM(_misread_extractor)

        transcript = await build_invoice_cove(extractor).run(_SOURCE)

        assert transcript.disposition is Disposition.SETTLED
        first, second = transcript.hearings
        assert [item.testimony for item in first.evidence.answered] == [True, False]
        assert [item.testimony for item in second.evidence.answered] == [True, True]
        assert transcript.ruling.invoice.total == Decimal("1234.50")
        assert [call.response_model for call in extractor.calls] == [Invoice, TotalCorrection]

    @pytest.mark.asyncio
    async def test_a_field_that_cannot_be_checked_is_flagged_for_review(self) -> None:
        async def registry_that_hangs_on_vendors(brief: WitnessBrief[InvoiceText, FieldClaim]) -> bool:
            if brief.challenge.field is InvoiceField.VENDOR:
                await wait_forever()
            return await grounded_in_source(brief)

        limits = Limits(witness_timeout_s=0.05, quorum=0.5)
        cove = build_invoice_cove(
            ScriptedLLM(_misread_extractor), witness=registry_that_hangs_on_vendors, limits=limits
        )

        transcript = await cove.run(_SOURCE)

        (unchecked,) = transcript.hearings[0].evidence.unanswered
        assert unchecked.reason is UnansweredReason.TIMEOUT
        assert transcript.disposition is Disposition.SETTLED
        assert transcript.ruling.needs_review == (InvoiceField.VENDOR,)
        assert transcript.ruling.invoice.total == Decimal("1234.50")
