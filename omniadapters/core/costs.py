from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field

from omniadapters.core.enums import Model, Provider

if TYPE_CHECKING:
    from omniadapters.core.models import CompletionUsage


class ModelPricing(BaseModel):
    input_cost_per_million: Decimal = Field(ge=Decimal(0))
    output_cost_per_million: Decimal = Field(ge=Decimal(0))
    provider: Provider

    model_config = ConfigDict(frozen=True)


class UsageCost(BaseModel):
    input_cost: Decimal
    output_cost: Decimal

    model_config = ConfigDict(frozen=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost(self) -> Decimal:
        return self.input_cost + self.output_cost


class CostResult(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: UsageCost
    model: Model | str

    model_config = ConfigDict(frozen=True)


MILLION = Decimal(1_000_000)

MODEL_PRICING_REGISTRY: dict[Model, ModelPricing] = {
    Model.GPT_4O: ModelPricing(
        input_cost_per_million=Decimal("2.50"),
        output_cost_per_million=Decimal("10.00"),
        provider=Provider.OPENAI,
    ),
    Model.GPT_4O_MINI: ModelPricing(
        input_cost_per_million=Decimal("0.15"),
        output_cost_per_million=Decimal("0.60"),
        provider=Provider.OPENAI,
    ),
    Model.O3_MINI: ModelPricing(
        input_cost_per_million=Decimal("1.10"),
        output_cost_per_million=Decimal("4.40"),
        provider=Provider.OPENAI,
    ),
    Model.O4_MINI: ModelPricing(
        input_cost_per_million=Decimal("1.10"),
        output_cost_per_million=Decimal("4.40"),
        provider=Provider.OPENAI,
    ),
    Model.CLAUDE_SONNET_4_5: ModelPricing(
        input_cost_per_million=Decimal("3.00"),
        output_cost_per_million=Decimal("15.00"),
        provider=Provider.ANTHROPIC,
    ),
    Model.CLAUDE_OPUS_4_5: ModelPricing(
        input_cost_per_million=Decimal("15.00"),
        output_cost_per_million=Decimal("75.00"),
        provider=Provider.ANTHROPIC,
    ),
    Model.CLAUDE_HAIKU_4_5: ModelPricing(
        input_cost_per_million=Decimal("1.00"),
        output_cost_per_million=Decimal("5.00"),
        provider=Provider.ANTHROPIC,
    ),
    Model.GEMINI_2_5_PRO: ModelPricing(
        input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("10.00"),
        provider=Provider.GEMINI,
    ),
    Model.GEMINI_2_5_FLASH: ModelPricing(
        input_cost_per_million=Decimal("0.15"),
        output_cost_per_million=Decimal("0.60"),
        provider=Provider.GEMINI,
    ),
    Model.GEMINI_2_5_FLASH_LITE: ModelPricing(
        input_cost_per_million=Decimal("0.10"),
        output_cost_per_million=Decimal("0.40"),
        provider=Provider.GEMINI,
    ),
}


class ModelPricingRegistry:
    def __init__(self, *, custom_pricing: dict[Model, ModelPricing] | None = None) -> None:
        self._registry: dict[Model, ModelPricing] = {**MODEL_PRICING_REGISTRY}
        if custom_pricing:
            self._registry.update(custom_pricing)

    def get(self, model: Model | str) -> ModelPricing | None:
        if isinstance(model, Model):
            return self._registry.get(model)

        try:
            return self._registry[Model(model)]
        except ValueError:
            pass

        for known_model, pricing in self._registry.items():
            if known_model.value in model or model in known_model.value:
                return pricing
        return None

    def register(self, model: Model, pricing: ModelPricing) -> None:
        self._registry[model] = pricing

    def list_models(self, *, provider: Provider | None = None) -> list[Model]:
        if provider is None:
            return list(self._registry.keys())
        return [m for m, p in self._registry.items() if p.provider == provider]


@lru_cache(maxsize=1)
def get_default_registry() -> ModelPricingRegistry:
    return ModelPricingRegistry()


def calculate_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    model: Model | str,
    registry: ModelPricingRegistry | None = None,
) -> UsageCost | None:
    registry = registry or get_default_registry()
    pricing = registry.get(model)

    if pricing is None:
        return None

    input_cost = (Decimal(prompt_tokens) / MILLION) * pricing.input_cost_per_million
    output_cost = (Decimal(completion_tokens) / MILLION) * pricing.output_cost_per_million

    return UsageCost(input_cost=input_cost, output_cost=output_cost)


def calculate_cost_from_usage(
    usage: CompletionUsage,
    model: Model | str,
    *,
    registry: ModelPricingRegistry | None = None,
) -> UsageCost | None:
    return calculate_cost(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        model=model,
        registry=registry,
    )


class CostTracker:
    def __init__(self, *, registry: ModelPricingRegistry | None = None) -> None:
        self._registry = registry or get_default_registry()
        self._results: list[CostResult] = []

    def track(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: Model | str,
    ) -> CostResult | None:
        cost = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            registry=self._registry,
        )

        if cost is None:
            return None

        result = CostResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            model=model,
        )
        self._results.append(result)
        return result

    def track_usage(self, usage: CompletionUsage, model: Model | str) -> CostResult | None:
        return self.track(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=model,
        )

    @property
    def results(self) -> list[CostResult]:
        return self._results.copy()

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self._results)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self._results)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._results)

    @property
    def total_input_cost(self) -> Decimal:
        return sum((r.cost.input_cost for r in self._results), Decimal(0))

    @property
    def total_output_cost(self) -> Decimal:
        return sum((r.cost.output_cost for r in self._results), Decimal(0))

    @property
    def total_cost(self) -> Decimal:
        return self.total_input_cost + self.total_output_cost

    @property
    def request_count(self) -> int:
        return len(self._results)

    def clear(self) -> None:
        self._results.clear()

    def __add__(self, other: Self) -> Self:
        combined = CostTracker(registry=self._registry)
        combined._results = self._results.copy() + other._results.copy()
        return combined  # type: ignore[return-value]
