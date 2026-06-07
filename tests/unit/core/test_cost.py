from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from genai_prices.types import PriceCalculation
from openai.types.chat import ChatCompletion
from pydantic_ai.usage import RequestUsage

from omniadapters.core.cost import (
    CostAccumulator,
    Unpriced,
    compute_cost,
)
from omniadapters.core.models import CompletionResponse

if TYPE_CHECKING:
    import pytest


def _raw_response(model: str) -> ChatCompletion:
    # raw_response is bound to a real provider response type; a minimal but valid
    # ChatCompletion satisfies the bound for these foundation-layer tests.
    return ChatCompletion(
        id="cmpl-test",
        choices=[],
        created=0,
        model=model,
        object="chat.completion",
    )


def _response(
    *,
    model: str,
    provider_id: str,
    usage: RequestUsage | None,
) -> CompletionResponse[ChatCompletion]:
    return CompletionResponse(
        content="hello",
        model=model,
        provider_id=provider_id,
        usage=usage,
        raw_response=_raw_response(model),
    )


class TestPydanticRequestUsageInterop:
    """A pydantic v2 BaseModel must hold a stdlib ``RequestUsage`` dataclass.

    This is the highest-risk interop point. Verified empirically: with
    ``arbitrary_types_allowed=True`` pydantic accepts the dataclass instance
    as-is (same object identity round-trips) in a ``RequestUsage | None`` field,
    and serializes it to a plain dict in both python and json modes.
    """

    def test_basemodel_holds_request_usage_instance(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)

        expected_total_tokens = 1500
        assert response.usage is usage
        assert response.usage is not None
        assert response.usage.total_tokens == expected_total_tokens

    def test_basemodel_holds_none_usage(self) -> None:
        response = _response(model="gpt-4o", provider_id="openai", usage=None)
        assert response.usage is None

    def test_model_dump_serializes_usage_dataclass(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)

        expected_input_tokens = 1000
        expected_output_tokens = 500
        dumped = response.model_dump()
        assert dumped["usage"]["input_tokens"] == expected_input_tokens
        assert dumped["usage"]["output_tokens"] == expected_output_tokens
        # raw_response is excluded; cost is a cached_property, not a field.
        assert "raw_response" not in dumped
        assert "cost" not in dumped

    def test_model_dump_json_mode_does_not_choke(self) -> None:
        expected_input_tokens = 10
        usage = RequestUsage(input_tokens=expected_input_tokens, output_tokens=20)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)
        dumped = response.model_dump(mode="json")
        assert dumped["usage"]["input_tokens"] == expected_input_tokens


class TestCachedPropertyCost:
    def test_cost_computed_once_and_cached(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)

        first = response.cost
        second = response.cost
        # cached_property returns the identical object on the second read.
        assert first is second

    def test_cost_is_not_a_serialized_field(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)

        # Reading cost first, then dumping, must not raise nor leak `cost`.
        _ = response.cost
        dumped = response.model_dump()
        assert "cost" not in dumped

    def test_cost_returns_price_calculation_for_priced_call(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        response = _response(model="gpt-4o", provider_id="openai", usage=usage)

        cost = response.cost
        assert isinstance(cost, PriceCalculation)
        assert cost.total_price > 0


class TestComputeCostHappyPath:
    def test_gpt_4o_priced(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        result = compute_cost(usage, model_ref="gpt-4o", provider_id="openai")

        assert isinstance(result, PriceCalculation)
        assert result.total_price > 0


class TestComputeCostUnpriced:
    def test_no_usage_when_none(self) -> None:
        result = compute_cost(None, model_ref="gpt-4o", provider_id="openai")
        assert isinstance(result, Unpriced)
        assert result.kind == "no_usage"

    def test_no_usage_when_all_zero(self) -> None:
        usage = RequestUsage()  # has_values() is False
        result = compute_cost(usage, model_ref="gpt-4o", provider_id="openai")
        assert isinstance(result, Unpriced)
        assert result.kind == "no_usage"

    def test_unknown_model(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        result = compute_cost(
            usage,
            model_ref="totally-made-up-model-xyz",
            provider_id="openai",
        )
        assert isinstance(result, Unpriced)
        assert result.kind == "unknown_model"
        assert "model" in result.detail.lower()

    def test_unknown_provider(self) -> None:
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        result = compute_cost(
            usage,
            model_ref="gpt-4o",
            provider_id="not-a-provider",
        )
        assert isinstance(result, Unpriced)
        assert result.kind == "unknown_provider"
        assert "provider" in result.detail.lower()

    def test_pricing_error_on_inconsistent_buckets(self) -> None:
        # cache_read_tokens exceeds input_tokens -> genai-prices raises ValueError
        # ("Uncached text input tokens cannot be negative").
        usage = RequestUsage(input_tokens=100, cache_read_tokens=1000, output_tokens=500)
        result = compute_cost(usage, model_ref="gpt-4o", provider_id="openai")
        assert isinstance(result, Unpriced)
        assert result.kind == "pricing_error"

    def test_pricing_error_via_monkeypatched_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> PriceCalculation:
            msg = "synthetic pricing failure"
            raise ValueError(msg)

        monkeypatch.setattr("omniadapters.core.cost.calc_price", _boom)
        usage = RequestUsage(input_tokens=1000, output_tokens=500)
        result = compute_cost(usage, model_ref="gpt-4o", provider_id="openai")
        assert isinstance(result, Unpriced)
        assert result.kind == "pricing_error"
        assert result.detail == "synthetic pricing failure"


class TestCostAccumulator:
    def test_mixed_model_session(self) -> None:
        # Confirmed present in the bundled genai-prices catalog:
        #   openai/gpt-4o, anthropic/claude-sonnet-4-6, google/gemini-2.5-flash.
        responses = [
            _response(
                model="gpt-4o",
                provider_id="openai",
                usage=RequestUsage(input_tokens=1000, output_tokens=500),
            ),
            _response(
                model="claude-sonnet-4-6",
                provider_id="anthropic",
                usage=RequestUsage(input_tokens=2000, output_tokens=1000),
            ),
            _response(
                model="gemini-2.5-flash",
                provider_id="google",
                usage=RequestUsage(input_tokens=3000, output_tokens=1500),
            ),
        ]

        accumulator = CostAccumulator()
        results = [accumulator.track(response) for response in responses]

        expected_request_count = 3
        assert all(isinstance(result, PriceCalculation) for result in results)
        assert accumulator.request_count == expected_request_count
        assert accumulator.total_tokens == 1000 + 500 + 2000 + 1000 + 3000 + 1500

        breakdown = accumulator.per_model_breakdown()
        assert set(breakdown.keys()) == {"gpt-4o", "claude-sonnet-4-6", "gemini-2.5-flash"}
        assert sum(breakdown.values()) == accumulator.total_cost
        assert accumulator.total_cost > 0
        assert accumulator.unpriced_calls == []

    def test_track_returns_cost_result(self) -> None:
        accumulator = CostAccumulator()
        response = _response(
            model="gpt-4o",
            provider_id="openai",
            usage=RequestUsage(input_tokens=1000, output_tokens=500),
        )
        result = accumulator.track(response)
        assert isinstance(result, PriceCalculation)

    def test_track_reuses_response_cost_object(self) -> None:
        accumulator = CostAccumulator()
        response = _response(
            model="gpt-4o",
            provider_id="openai",
            usage=RequestUsage(input_tokens=1000, output_tokens=500),
        )
        tracked = accumulator.track(response)
        # track() must reuse the cached_property, not recompute.
        assert tracked is response.cost

    def test_unpriced_call_recorded(self) -> None:
        accumulator = CostAccumulator()
        response = _response(
            model="totally-made-up-model-xyz",
            provider_id="openai",
            usage=RequestUsage(input_tokens=1000, output_tokens=500),
        )
        result = accumulator.track(response)

        expected_total_tokens = 1500
        assert isinstance(result, Unpriced)
        assert result.kind == "unknown_model"
        assert len(accumulator.unpriced_calls) == 1
        assert accumulator.unpriced_calls[0].kind == "unknown_model"
        # Tokens are still aggregated even when the call is unpriced.
        assert accumulator.total_tokens == expected_total_tokens
        assert accumulator.request_count == 1
        assert accumulator.total_cost == Decimal(0)

    def test_track_none_usage_still_counts_request(self) -> None:
        accumulator = CostAccumulator()
        response = _response(model="gpt-4o", provider_id="openai", usage=None)
        result = accumulator.track(response)

        assert isinstance(result, Unpriced)
        assert result.kind == "no_usage"
        # No usage to incr, but the request itself is still tracked.
        assert accumulator.request_count == 1
        assert accumulator.total_tokens == 0

    def test_clear_resets_state(self) -> None:
        accumulator = CostAccumulator()
        accumulator.track(
            _response(
                model="gpt-4o",
                provider_id="openai",
                usage=RequestUsage(input_tokens=1000, output_tokens=500),
            ),
        )
        accumulator.clear()

        assert accumulator.request_count == 0
        assert accumulator.total_tokens == 0
        assert accumulator.total_cost == Decimal(0)
        assert accumulator.per_model_breakdown() == {}
        assert accumulator.unpriced_calls == []

    def test_add_combines_accumulators(self) -> None:
        first = CostAccumulator()
        first.track(
            _response(
                model="gpt-4o",
                provider_id="openai",
                usage=RequestUsage(input_tokens=1000, output_tokens=500),
            ),
        )

        second = CostAccumulator()
        second.track(
            _response(
                model="claude-sonnet-4-6",
                provider_id="anthropic",
                usage=RequestUsage(input_tokens=2000, output_tokens=1000),
            ),
        )

        combined = first + second

        expected_request_count = 2
        assert combined.request_count == expected_request_count
        assert combined.total_tokens == 1000 + 500 + 2000 + 1000
        assert combined.total_cost == first.total_cost + second.total_cost
        assert set(combined.per_model_breakdown().keys()) == {"gpt-4o", "claude-sonnet-4-6"}
        # Operands are not mutated by __add__.
        assert first.request_count == 1
        assert second.request_count == 1

    def test_add_merges_per_model_for_same_model(self) -> None:
        first = CostAccumulator()
        first.track(
            _response(
                model="gpt-4o",
                provider_id="openai",
                usage=RequestUsage(input_tokens=1000, output_tokens=500),
            ),
        )
        second = CostAccumulator()
        second.track(
            _response(
                model="gpt-4o",
                provider_id="openai",
                usage=RequestUsage(input_tokens=1000, output_tokens=500),
            ),
        )

        combined = first + second
        breakdown = combined.per_model_breakdown()
        assert set(breakdown.keys()) == {"gpt-4o"}
        assert breakdown["gpt-4o"] == first.total_cost + second.total_cost
