"""OpenAI tool-calling smoke-test demo.

The model call is real. The local tool performs deterministic arithmetic so the
demo does not depend on fake fixture data or a second external service.

Examples:
```bash
uv run playground/completion/02_tools_complete.py
uv run playground/completion/02_tools_complete.py --stream
```

"""

from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal, Self

from _shared import (
    JsonSchema,
    PlaygroundConfigurationError,
    console,
    create_openai_adapter,
    model_json_schema,
    render_completion_error,
    render_configuration_error,
    render_unexpected_error,
    response_panel,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError

if TYPE_CHECKING:
    from omniadapters.completion.adapters.openai import OpenAIAdapter
    from omniadapters.core.types import MessageParam

ToolChoice = Literal["auto"] | dict[str, object]


class CompletionCostRequest(BaseModel):
    """Arguments accepted by the local completion-cost tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: Annotated[int, Field(gt=0, description="Number of prompt/input tokens.")]
    output_tokens: Annotated[int, Field(gt=0, description="Number of completion/output tokens.")]
    input_price_per_million: Annotated[
        Decimal,
        Field(gt=Decimal(0), description="Input-token price in USD per 1,000,000 tokens."),
    ]
    output_price_per_million: Annotated[
        Decimal,
        Field(gt=Decimal(0), description="Output-token price in USD per 1,000,000 tokens."),
    ]

    @classmethod
    def from_json(cls: type[Self], raw: str) -> Self:
        """Validate tool-call JSON arguments."""
        return cls.model_validate_json(raw)


class CompletionCostResult(BaseModel):
    """Result returned by the local completion-cost tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int
    output_tokens: int
    input_cost_usd: Decimal
    output_cost_usd: Decimal
    total_cost_usd: Decimal


class ToolFunctionCall(BaseModel):
    """Portable function-call shape used in assistant messages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    arguments: str


class ToolCallRequest(BaseModel):
    """Portable tool-call shape used in assistant and tool messages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class StreamedToolCall(BaseModel):
    """Mutable accumulator for one streamed OpenAI tool call."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    type: Literal["function"] = "function"
    name: str = ""
    arguments: str = ""

    def append_delta(self, delta: dict[str, object]) -> None:
        """Merge one OpenAI streaming tool-call delta into this accumulator."""
        if tool_call_id := delta.get("id"):
            self.id = str(tool_call_id)
        if tool_call_type := delta.get("type"):
            self.type = "function" if str(tool_call_type) == "function" else self.type

        function = delta.get("function")
        if not isinstance(function, dict):
            return

        if name := function.get("name"):
            self.name += str(name)
        if arguments := function.get("arguments"):
            self.arguments += str(arguments)

    def to_request(self) -> ToolCallRequest:
        """Convert accumulated deltas into a complete tool-call request."""
        return ToolCallRequest(
            id=self.id,
            type=self.type,
            function=ToolFunctionCall(name=self.name, arguments=self.arguments),
        )


class ToolCallAccumulator:
    """Collect OpenAI streaming tool-call deltas by index."""

    def __init__(self) -> None:
        self._calls: dict[int, StreamedToolCall] = {}

    def add_delta(self, delta: dict[str, object]) -> None:
        """Add one streamed tool-call delta."""
        index = delta.get("index")
        if not isinstance(index, int):
            return

        call = self._calls.setdefault(index, StreamedToolCall())
        call.append_delta(delta)

    def complete_calls(self) -> list[ToolCallRequest]:
        """Return completed tool calls in stream order."""
        return [self._calls[index].to_request() for index in sorted(self._calls)]


def calculate_completion_cost(request: CompletionCostRequest) -> CompletionCostResult:
    """Calculate exact token-cost arithmetic from validated tool arguments."""
    divisor = Decimal(1_000_000)
    input_cost = (Decimal(request.input_tokens) / divisor) * request.input_price_per_million
    output_cost = (Decimal(request.output_tokens) / divisor) * request.output_price_per_million
    return CompletionCostResult(
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


def completion_cost_tool() -> dict[str, object]:
    """Return the OpenAI tool schema for completion-cost calculation."""
    parameters: JsonSchema = model_json_schema(CompletionCostRequest)
    return {
        "type": "function",
        "function": {
            "name": "calculate_completion_cost",
            "description": "Calculate exact USD cost from token counts and per-million token prices.",
            "parameters": parameters,
        },
    }


def forced_completion_cost_tool_choice() -> ToolChoice:
    """Force OpenAI to exercise the demo tool during smoke tests."""
    return {"type": "function", "function": {"name": "calculate_completion_cost"}}


def print_cost_result(result: CompletionCostResult) -> None:
    """Render cost-tool output."""
    table = Table(title="Local tool result", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Input tokens", str(result.input_tokens))
    table.add_row("Output tokens", str(result.output_tokens))
    table.add_row("Input cost", f"${result.input_cost_usd:.8f}")
    table.add_row("Output cost", f"${result.output_cost_usd:.8f}")
    table.add_row("Total cost", f"${result.total_cost_usd:.8f}")
    console.print(table)


def execute_tool_call(tool_call: ToolCallRequest) -> CompletionCostResult:
    """Execute a validated tool call from the model."""
    if tool_call.function.name != "calculate_completion_cost":
        msg = f"Unsupported tool requested: {tool_call.function.name}"
        raise PlaygroundConfigurationError(msg)

    request = CompletionCostRequest.from_json(tool_call.function.arguments)
    return calculate_completion_cost(request)


def assistant_tool_message(tool_calls: list[ToolCallRequest]) -> MessageParam:
    """Create the assistant message that records model-requested tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call.model_dump(mode="json") for tool_call in tool_calls],
    }


def tool_result_message(tool_call: ToolCallRequest, result: CompletionCostResult) -> MessageParam:
    """Create the tool-result message sent back to OpenAI."""
    return {
        "role": "tool",
        "content": result.model_dump_json(),
        "tool_call_id": tool_call.id,
    }


async def collect_initial_tool_calls(
    adapter: OpenAIAdapter,
    messages: list[MessageParam],
) -> list[ToolCallRequest]:
    """Ask OpenAI for non-streaming tool calls."""
    response = await adapter.agenerate(
        messages=messages,
        tools=[completion_cost_tool()],
        tool_choice=forced_completion_cost_tool_choice(),
        temperature=0.0,
    )
    raw_response = response.raw_response
    choice = raw_response.choices[0] if raw_response.choices else None
    tool_calls = choice.message.tool_calls if choice and choice.message.tool_calls else []

    requests: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        if tool_call.type != "function":
            continue
        requests.append(
            ToolCallRequest(
                id=tool_call.id,
                type="function",
                function=ToolFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
        )
    return requests


async def collect_streamed_tool_calls(
    adapter: OpenAIAdapter,
    messages: list[MessageParam],
) -> list[ToolCallRequest]:
    """Ask OpenAI for streaming tool calls and accumulate deltas."""
    accumulator = ToolCallAccumulator()
    stream = await adapter.agenerate(
        messages=messages,
        stream=True,
        tools=[completion_cost_tool()],
        tool_choice=forced_completion_cost_tool_choice(),
        temperature=0.0,
    )

    async for chunk in stream:
        if not chunk.tool_calls:
            continue
        for delta in chunk.tool_calls:
            accumulator.add_delta(delta)

    return accumulator.complete_calls()


async def complete_tool_call_flow(*, stream: bool) -> None:
    """Run the full OpenAI tool-calling flow."""
    adapter = create_openai_adapter()
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                "Use the calculate_completion_cost tool for a request with 1200 input tokens, "
                "350 output tokens, input price 0.15 USD per million tokens, and output price "
                "0.60 USD per million tokens. Then summarize the result in one sentence."
            ),
        }
    ]

    try:
        console.print(
            Panel.fit(
                "OpenAI tool-calling smoke test\n[dim]Real model call, deterministic local tool[/dim]",
                border_style="cyan",
            )
        )
        console.print(f"[yellow]User:[/yellow] {messages[0]['content']}")

        if stream:
            tool_calls = await collect_streamed_tool_calls(adapter, messages)
        else:
            tool_calls = await collect_initial_tool_calls(adapter, messages)

        if not tool_calls:
            message = "OpenAI returned no tool calls."
            raise PlaygroundConfigurationError(message)

        console.print(Syntax(json.dumps([call.model_dump(mode="json") for call in tool_calls], indent=2), "json"))
        messages.append(assistant_tool_message(tool_calls))

        for tool_call in tool_calls:
            result = execute_tool_call(tool_call)
            print_cost_result(result)
            messages.append(tool_result_message(tool_call, result))

        final_response = await adapter.agenerate(messages=messages, temperature=0.0)
        console.print(response_panel(final_response, "openai"))
    finally:
        await adapter.aclose()


async def main() -> None:
    """Parse CLI options and run the OpenAI tool-calling demo."""
    parser = argparse.ArgumentParser(description="OpenAI tool-calling smoke-test demo")
    parser.add_argument("--stream", action="store_true", help="Collect the first tool call via streaming chunks")
    args = parser.parse_args()

    try:
        await complete_tool_call_flow(stream=args.stream)
    except CompletionHTTPError as error:
        render_completion_error(error)
    except CompletionAPIError as error:
        render_completion_error(error)
    except PlaygroundConfigurationError as error:
        render_configuration_error(error)
    except ValidationError as error:
        console.print(
            Panel(
                Syntax(error.json(indent=2), "json", theme="monokai", word_wrap=True),
                title="Tool argument validation error",
                border_style="red",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tool-calling demo interrupted.[/yellow]")
    except Exception as error:  # noqa: BLE001 - CLI boundary renders unexpected smoke-test failures.
        render_unexpected_error(error)
