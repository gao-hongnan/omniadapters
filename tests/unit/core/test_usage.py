from __future__ import annotations

from decimal import Decimal

import pytest

from omniadapters.core.enums import Model, Provider
from omniadapters.core.models import Usage
from omniadapters.core.usage import (
    ModelPricing,
    ModelPricingRegistry,
    UsageCost,
    UsageSnapshot,
    UsageTracker,
    compute_cost,
    compute_cost_from_usage,
    get_default_registry,
)


class TestModelPricing:
    def test_create_valid_pricing(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million=Decimal("2.50"),
            output_cost_per_million=Decimal("10.00"),
            provider=Provider.OPENAI,
        )
        assert pricing.input_cost_per_million == Decimal("2.50")
        assert pricing.output_cost_per_million == Decimal("10.00")
        assert pricing.provider == Provider.OPENAI

    def test_pricing_is_immutable(self) -> None:
        pricing = ModelPricing(
            input_cost_per_million=Decimal("2.50"),
            output_cost_per_million=Decimal("10.00"),
            provider=Provider.OPENAI,
        )
        with pytest.raises((TypeError, ValueError)):
            pricing.input_cost_per_million = Decimal("5.00")  # type: ignore[misc]

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError):
            ModelPricing(
                input_cost_per_million=Decimal("-1.00"),
                output_cost_per_million=Decimal("10.00"),
                provider=Provider.OPENAI,
            )


class TestUsageCost:
    def test_total_cost_computed(self) -> None:
        cost = UsageCost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
        )
        assert cost.total_cost == Decimal("0.003")

    def test_usage_cost_immutable(self) -> None:
        cost = UsageCost(
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.002"),
        )
        with pytest.raises((TypeError, ValueError)):
            cost.input_cost = Decimal("0.005")  # type: ignore[misc]


class TestModelPricingRegistry:
    def test_get_known_model_by_enum(self) -> None:
        registry = get_default_registry()
        pricing = registry.get(Model.GPT_4O)
        assert pricing is not None
        assert pricing.provider == Provider.OPENAI

    def test_get_known_model_by_string(self) -> None:
        registry = get_default_registry()
        pricing = registry.get("gpt-4o")
        assert pricing is not None
        assert pricing.provider == Provider.OPENAI

    def test_get_unknown_model_returns_none(self) -> None:
        registry = ModelPricingRegistry()
        pricing = registry.get("unknown-model-xyz")
        assert pricing is None

    def test_fuzzy_match_model(self) -> None:
        registry = get_default_registry()
        pricing = registry.get("gpt-4o-some-variant")
        assert pricing is not None

    def test_list_models_all(self) -> None:
        registry = get_default_registry()
        models = registry.list_models()
        assert len(models) > 0
        assert Model.GPT_4O in models

    def test_list_models_by_provider(self) -> None:
        registry = get_default_registry()
        openai_models = registry.list_models(provider=Provider.OPENAI)
        anthropic_models = registry.list_models(provider=Provider.ANTHROPIC)

        assert all("gpt" in m.value or "o3" in m.value or "o4" in m.value for m in openai_models)
        assert all("claude" in m.value for m in anthropic_models)


class TestComputeCost:
    def test_compute_cost_gpt4o(self) -> None:
        cost = compute_cost(
            input_tokens=1000,
            output_tokens=500,
            model=Model.GPT_4O,
        )
        assert cost is not None
        expected_input = Decimal(1000) / Decimal(1_000_000) * Decimal("2.50")
        expected_output = Decimal(500) / Decimal(1_000_000) * Decimal("10.00")
        assert cost.input_cost == expected_input
        assert cost.output_cost == expected_output
        assert cost.total_cost == expected_input + expected_output

    def test_compute_cost_unknown_model(self) -> None:
        cost = compute_cost(
            input_tokens=1000,
            output_tokens=500,
            model="unknown-model-xyz",
        )
        assert cost is None


class TestComputeCostFromUsage:
    def test_compute_from_usage(self) -> None:
        usage = Usage(
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
        )
        cost = compute_cost_from_usage(usage, Model.GPT_4O)
        assert cost is not None
        assert cost.total_cost > 0


class TestUsageTracker:
    def test_track_single_request(self) -> None:
        tracker = UsageTracker()
        result = tracker.track(
            input_tokens=1000,
            output_tokens=500,
            model=Model.GPT_4O,
        )
        assert result is not None
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500
        assert result.model == Model.GPT_4O

    def test_track_multiple_requests(self) -> None:
        tracker = UsageTracker()
        tracker.track(input_tokens=1000, output_tokens=500, model=Model.GPT_4O)
        tracker.track(input_tokens=2000, output_tokens=1000, model=Model.GPT_4O)

        assert tracker.request_count == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.total_tokens == 4500

    def test_track_unknown_model_returns_none(self) -> None:
        tracker = UsageTracker()
        result = tracker.track(
            input_tokens=1000,
            output_tokens=500,
            model="unknown-model",
        )
        assert result is None
        assert tracker.request_count == 0

    def test_track_usage_method(self) -> None:
        tracker = UsageTracker()
        usage = Usage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )
        result = tracker.track_usage(usage, Model.GPT_4O)
        assert result is not None
        assert tracker.request_count == 1

    def test_total_costs(self) -> None:
        tracker = UsageTracker()
        tracker.track(input_tokens=1_000_000, output_tokens=500_000, model=Model.GPT_4O)

        assert tracker.total_input_cost == Decimal("2.50")
        assert tracker.total_output_cost == Decimal("5.00")
        assert tracker.total_cost == Decimal("7.50")

    def test_clear_tracker(self) -> None:
        tracker = UsageTracker()
        tracker.track(input_tokens=1000, output_tokens=500, model=Model.GPT_4O)
        tracker.clear()

        assert tracker.request_count == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == Decimal(0)

    def test_results_property_returns_copy(self) -> None:
        tracker = UsageTracker()
        tracker.track(input_tokens=1000, output_tokens=500, model=Model.GPT_4O)

        results = tracker.results
        results.clear()

        assert tracker.request_count == 1

    def test_add_trackers(self) -> None:
        tracker1 = UsageTracker()
        tracker1.track(input_tokens=1000, output_tokens=500, model=Model.GPT_4O)

        tracker2 = UsageTracker()
        tracker2.track(input_tokens=2000, output_tokens=1000, model=Model.GPT_4O)

        combined = tracker1 + tracker2

        assert combined.request_count == 2
        assert combined.total_input_tokens == 3000
        assert combined.total_output_tokens == 1500


class TestUsageSnapshot:
    def test_usage_snapshot_immutable(self) -> None:
        result = UsageSnapshot(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost=UsageCost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
            model=Model.GPT_4O,
        )
        with pytest.raises((TypeError, ValueError)):
            result.input_tokens = 2000  # type: ignore[misc]
