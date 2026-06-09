"""Cost computation and aggregation over pydantic-ai usage and genai-prices.

This module owns the post-call cost vocabulary for omniadapters:

- :func:`compute_cost` turns a single :class:`~pydantic_ai.usage.RequestUsage`
  plus a ``(model_ref, provider_id)`` pair into a typed result that is either a
  :class:`~genai_prices.types.PriceCalculation` (priced) or an :class:`Unpriced`
  reason (no usage / unknown model / unknown provider / pricing error). Failure
  is part of the return type, never an exception, so callers exhaustively narrow.
- :class:`CostAccumulator` aggregates tokens (via a ``RunUsage``) and money
  (per-call :class:`PriceCalculation` records) across a multi-model session.
  Prices are summed per-call because :func:`genai_prices.calc_price` takes a
  single ``model_ref``; aggregating tokens first would misprice mixed sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self

from genai_prices import calc_price
from genai_prices.types import PriceCalculation
from pydantic_ai.usage import RequestUsage, RunUsage

from .enums import Provider

if TYPE_CHECKING:
    from datetime import datetime

    from .models import CompletionResponse


@dataclass(frozen=True, slots=True)
class UsageExtractionSpec:
    """The genai-prices coordinates needed to extract usage from a provider dump.

    These values are class-invariant per provider -- they never change between
    calls -- so they are declared once in :data:`GENAI_PRICES_PROFILE` rather than
    hardcoded inline in each adapter's ``_to_unified_response``.

    Attributes
    ----------
    provider
        The genai-prices provider id (e.g. ``"openai"``, ``"google"``). Stored on
        :attr:`CompletionResponse.provider_id` and passed to :func:`compute_cost`.
        Typed ``str`` deliberately: this is genai-prices' vocabulary, which differs
        from the omniadapters :class:`~omniadapters.core.enums.Provider` enum
        (Gemini -> ``"google"``, Azure -> ``"azure"``) and is intentionally NOT
        pinned to the upstream ``ProviderID`` literal, which can drift across
        genai-prices versions. The exact values are pinned by the registry test.
    provider_url
        Provider base URL genai-prices matches against its catalog ``api_pattern``
        to resolve the provider. Azure overrides this per-call with its configured
        endpoint.
    provider_fallback
        Fallback provider id used when ``provider`` cannot be resolved (Azure falls
        back to ``"openai"`` since it carries the OpenAI usage shape).
    api_flavor
        genai-prices extractor flavor. OpenAI/Azure require ``"chat"`` (they have
        no ``"default"`` extractor); Anthropic/Google use ``"default"``.
    by_alias
        Whether the SDK response must be dumped with ``by_alias=True`` for
        genai-prices to read it. Gemini's google catalog reads camelCase keys, so
        its snake_case dump yields an all-zero usage unless aliased.

    """

    provider: str
    provider_url: str
    provider_fallback: str
    api_flavor: str
    by_alias: bool = False


def extract_usage(raw_dump: dict[str, Any], spec: UsageExtractionSpec) -> RequestUsage:
    """Extract a :class:`RequestUsage` from a provider response dump via ``spec``.

    Single owner of the :meth:`RequestUsage.extract` call mechanics; adapters
    pass the dump they produced (using ``spec.by_alias`` to choose the dump mode)
    plus their spec, instead of re-typing the genai-prices argument tuple.
    """
    return RequestUsage.extract(
        raw_dump,
        provider=spec.provider,
        provider_url=spec.provider_url,
        provider_fallback=spec.provider_fallback,
        api_flavor=spec.api_flavor,
    )


GENAI_PRICES_PROFILE: dict[Provider, UsageExtractionSpec] = {
    Provider.OPENAI: UsageExtractionSpec(
        provider="openai",
        provider_url="https://api.openai.com",
        provider_fallback="openai",
        api_flavor="chat",
    ),
    Provider.ANTHROPIC: UsageExtractionSpec(
        provider="anthropic",
        provider_url="https://api.anthropic.com",
        provider_fallback="anthropic",
        api_flavor="default",
    ),
    Provider.GEMINI: UsageExtractionSpec(
        provider="google",
        provider_url="https://generativelanguage.googleapis.com",
        provider_fallback="google",
        api_flavor="default",
        by_alias=True,
    ),
    Provider.AZURE_OPENAI: UsageExtractionSpec(
        provider="azure",
        provider_url="https://api.openai.com",
        provider_fallback="openai",
        api_flavor="chat",
    ),
}
"""Single source of truth mapping each :class:`Provider` to its genai-prices spec."""


type UnpricedKind = Literal["no_usage", "unknown_model", "unknown_provider", "pricing_error"]


@dataclass(frozen=True, slots=True)
class Unpriced:
    """A call that could not be priced, with a discriminator and human-readable detail."""

    kind: UnpricedKind
    detail: str


type CostResult = PriceCalculation | Unpriced
"""The result of pricing a single call: a concrete price or a typed reason it has none."""


def compute_cost(
    usage: RequestUsage | None,
    *,
    model_ref: str,
    provider_id: str,
    timestamp: datetime | None = None,
) -> CostResult:
    """Price a single model call.

    Parameters
    ----------
    usage
        The per-call usage. ``None`` (or an all-zero usage) yields ``"no_usage"``.
    model_ref
        The provider-native model identifier (e.g. ``"gpt-4o"``).
    provider_id
        The genai-prices provider id (e.g. ``"openai"``, ``"anthropic"``, ``"google"``).
    timestamp
        Optional request timestamp for historical pricing snapshots.

    Returns
    -------
    CostResult
        A :class:`PriceCalculation` on success, otherwise an :class:`Unpriced`
        carrying one of the :data:`UnpricedKind` discriminators.

    """
    if usage is None or not usage.has_values():
        return Unpriced(kind="no_usage", detail="usage is missing or has no token values")

    try:
        return calc_price(
            usage,
            model_ref,
            provider_id=provider_id,
            genai_request_timestamp=timestamp,
        )
    except LookupError as error:
        message = str(error)
        if message.startswith("Unable to find provider"):
            return Unpriced(kind="unknown_provider", detail=message)
        return Unpriced(kind="unknown_model", detail=message)
    except ValueError as error:
        return Unpriced(kind="pricing_error", detail=str(error))


@dataclass(slots=True)
class CostAccumulator:
    """Aggregate token usage and per-call cost across a multi-model session.

    Tokens accumulate into a single ``RunUsage`` (addable across providers), while
    money is summed per-call into :class:`PriceCalculation` records and a per-model
    ``Decimal`` map. This preserves the distinction pydantic-ai's types make: tokens
    are addable, but prices need per-call ``model_ref`` context.
    """

    _run_usage: RunUsage = field(default_factory=RunUsage)
    _priced: list[PriceCalculation] = field(default_factory=list)
    _unpriced: list[Unpriced] = field(default_factory=list)
    _per_model: dict[str, Decimal] = field(default_factory=dict)

    def track(self, response: CompletionResponse[Any]) -> CostResult:
        """Record one response: aggregate its tokens and its cost.

        Reuses ``response.cost`` (the cached property) rather than recomputing.
        The request is always counted, even when usage is absent or unpriced.
        """
        if response.usage is not None:
            self._run_usage.incr(response.usage)
        # RunUsage.incr does not bump `requests` for a RequestUsage argument, so
        # count each tracked call manually (one CompletionResponse == one request).
        self._run_usage.requests += 1

        result = response.cost
        if isinstance(result, PriceCalculation):
            self._priced.append(result)
            self._per_model[response.model] = self._per_model.get(response.model, Decimal(0)) + result.total_price
        else:
            self._unpriced.append(result)
        return result

    @property
    def total_cost(self) -> Decimal:
        return sum((calculation.total_price for calculation in self._priced), Decimal(0))

    @property
    def total_tokens(self) -> int:
        return self._run_usage.total_tokens

    @property
    def request_count(self) -> int:
        return self._run_usage.requests

    def per_model_breakdown(self) -> dict[str, Decimal]:
        """Return a copy of the per-model cost totals, keyed by ``response.model``."""
        return dict(self._per_model)

    @property
    def unpriced_calls(self) -> list[Unpriced]:
        return list(self._unpriced)

    def clear(self) -> None:
        self._run_usage = RunUsage()
        self._priced.clear()
        self._unpriced.clear()
        self._per_model.clear()

    def __add__(self, other: Self) -> Self:
        combined = type(self)()
        combined._run_usage = self._run_usage + other._run_usage
        combined._priced = [*self._priced, *other._priced]
        combined._unpriced = [*self._unpriced, *other._unpriced]
        for model, cost in (*self._per_model.items(), *other._per_model.items()):
            combined._per_model[model] = combined._per_model.get(model, Decimal(0)) + cost
        return combined


__all__ = [
    "GENAI_PRICES_PROFILE",
    "CostAccumulator",
    "CostResult",
    "Unpriced",
    "UnpricedKind",
    "UsageExtractionSpec",
    "compute_cost",
    "extract_usage",
]
