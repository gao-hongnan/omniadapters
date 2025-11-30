from __future__ import annotations

from decimal import Decimal

import pytest

from omniadapters.core.costs import (
    CostResult,
    CostTracker,
    ModelPricing,
    ModelPricingRegistry,
    UsageCost,
    calculate_cost,
    calculate_cost_from_usage,
    get_default_registry,
)
from omniadapters.core.enums import Model, Provider
from omniadapters.core.models import CompletionUsage


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

    def test_custom_pricing_override(self) -> None:
        custom = {
            Model.GPT_4O: ModelPricing(
                input_cost_per_million=Decimal("1.00"),
                output_cost_per_million=Decimal("2.00"),
                provider=Provider.OPENAI,
            )
        }
        registry = ModelPricingRegistry(custom_pricing=custom)
        pricing = registry.get(Model.GPT_4O)
        assert pricing is not None
        assert pricing.input_cost_per_million == Decimal("1.00")

    def test_register_new_model(self) -> None:
        registry = ModelPricingRegistry()
        new_pricing = ModelPricing(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("10.00"),
            provider=Provider.ANTHROPIC,
        )
        registry.register(Model.CLAUDE_SONNET_4_5, new_pricing)
        pricing = registry.get(Model.CLAUDE_SONNET_4_5)
        assert pricing is not None
        assert pricing.input_cost_per_million == Decimal("5.00")

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


class TestCalculateCost:
    def test_calculate_cost_gpt4o(self) -> None:
        cost = calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model=Model.GPT_4O,
        )
        assert cost is not None
        expected_input = Decimal(1000) / Decimal(1_000_000) * Decimal("2.50")
        expected_output = Decimal(500) / Decimal(1_000_000) * Decimal("10.00")
        assert cost.input_cost == expected_input
        assert cost.output_cost == expected_output
        assert cost.total_cost == expected_input + expected_output

    def test_calculate_cost_unknown_model(self) -> None:
        cost = calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="unknown-model-xyz",
        )
        assert cost is None

    def test_calculate_cost_with_custom_registry(self) -> None:
        custom = {
            Model.GPT_4O_MINI: ModelPricing(
                input_cost_per_million=Decimal("1.00"),
                output_cost_per_million=Decimal("1.00"),
                provider=Provider.OPENAI,
            )
        }
        registry = ModelPricingRegistry(custom_pricing=custom)
        cost = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            model=Model.GPT_4O_MINI,
            registry=registry,
        )
        assert cost is not None
        assert cost.input_cost == Decimal("1.00")
        assert cost.output_cost == Decimal("1.00")
        assert cost.total_cost == Decimal("2.00")


class TestCalculateCostFromUsage:
    def test_calculate_from_usage(self) -> None:
        usage = CompletionUsage(
            prompt_tokens=2000,
            completion_tokens=1000,
            total_tokens=3000,
        )
        cost = calculate_cost_from_usage(usage, Model.GPT_4O)
        assert cost is not None
        assert cost.total_cost > 0


class TestCostTracker:
    def test_track_single_request(self) -> None:
        tracker = CostTracker()
        result = tracker.track(
            prompt_tokens=1000,
            completion_tokens=500,
            model=Model.GPT_4O,
        )
        assert result is not None
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 500
        assert result.total_tokens == 1500
        assert result.model == Model.GPT_4O

    def test_track_multiple_requests(self) -> None:
        tracker = CostTracker()
        tracker.track(prompt_tokens=1000, completion_tokens=500, model=Model.GPT_4O)
        tracker.track(prompt_tokens=2000, completion_tokens=1000, model=Model.GPT_4O)

        assert tracker.request_count == 2
        assert tracker.total_prompt_tokens == 3000
        assert tracker.total_completion_tokens == 1500
        assert tracker.total_tokens == 4500

    def test_track_unknown_model_returns_none(self) -> None:
        tracker = CostTracker()
        result = tracker.track(
            prompt_tokens=1000,
            completion_tokens=500,
            model="unknown-model",
        )
        assert result is None
        assert tracker.request_count == 0

    def test_track_usage_method(self) -> None:
        tracker = CostTracker()
        usage = CompletionUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        result = tracker.track_usage(usage, Model.GPT_4O)
        assert result is not None
        assert tracker.request_count == 1

    def test_total_costs(self) -> None:
        tracker = CostTracker()
        tracker.track(prompt_tokens=1_000_000, completion_tokens=500_000, model=Model.GPT_4O)

        assert tracker.total_input_cost == Decimal("2.50")
        assert tracker.total_output_cost == Decimal("5.00")
        assert tracker.total_cost == Decimal("7.50")

    def test_clear_tracker(self) -> None:
        tracker = CostTracker()
        tracker.track(prompt_tokens=1000, completion_tokens=500, model=Model.GPT_4O)
        tracker.clear()

        assert tracker.request_count == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == Decimal(0)

    def test_results_property_returns_copy(self) -> None:
        tracker = CostTracker()
        tracker.track(prompt_tokens=1000, completion_tokens=500, model=Model.GPT_4O)

        results = tracker.results
        results.clear()

        assert tracker.request_count == 1

    def test_add_trackers(self) -> None:
        tracker1 = CostTracker()
        tracker1.track(prompt_tokens=1000, completion_tokens=500, model=Model.GPT_4O)

        tracker2 = CostTracker()
        tracker2.track(prompt_tokens=2000, completion_tokens=1000, model=Model.GPT_4O)

        combined = tracker1 + tracker2

        assert combined.request_count == 2
        assert combined.total_prompt_tokens == 3000
        assert combined.total_completion_tokens == 1500


class TestCostResult:
    def test_cost_result_immutable(self) -> None:
        result = CostResult(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost=UsageCost(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
            model=Model.GPT_4O,
        )
        with pytest.raises((TypeError, ValueError)):
            result.prompt_tokens = 2000  # type: ignore[misc]
