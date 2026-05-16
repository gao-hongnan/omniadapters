from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agents.cove import CoVeCandidate, CoVeOrchestrator
from .config.settings import get_settings
from .types import BatchVerificationResult, UserQuery, VerificationResult


async def run_verification(
    user_query: UserQuery,
    env_file: str | None = None,
    yaml_file: str | None = None,
    console: Console | None = None,
    question_label: str = "Question",
) -> CoVeCandidate:
    """Run verification using CoVe (Chain-of-Verification)."""
    settings = get_settings(env_file=env_file, yaml_file=yaml_file)
    cove_config = settings.cove

    orchestrator = CoVeOrchestrator(config=cove_config, console=console, question_label=question_label)
    result = await orchestrator.aexecute(user_query)

    return result


async def run_batch_verification(
    user_queries: list[UserQuery],
    env_file: str | None = None,
    yaml_file: str | None = None,
    console: Console | None = None,
) -> BatchVerificationResult:
    """Run verification on multiple user queries concurrently."""
    if console is not None:
        console.rule("[bold cyan]Batch CoVe Verification")
        console.log(f"Starting {len(user_queries)} questions")

    tasks = [
        run_verification(
            user_query=user_query,
            env_file=env_file,
            yaml_file=yaml_file,
            console=console,
            question_label=f"Question {index}/{len(user_queries)}",
        )
        for index, user_query in enumerate(user_queries, start=1)
    ]
    results = await asyncio.gather(*tasks)

    verification_results = [
        VerificationResult(user_query=user_query, result=result)
        for user_query, result in zip(user_queries, results, strict=False)
    ]

    aligned_count = sum(1 for r in results if r.is_aligned)
    total_confidence = sum(r.confidence for r in results)
    if console is not None:
        console.log(f"Batch finished aligned={aligned_count}/{len(results)}")

    return BatchVerificationResult(
        results=verification_results,
        total_questions=len(user_queries),
        aligned_count=aligned_count,
        average_confidence=total_confidence / len(results) if results else 0.0,
    )


def load_questions(file_path: str | Path) -> list[UserQuery]:
    file_path = Path(file_path)
    if file_path.suffix != ".json":
        msg = f"Unsupported file format: {file_path.suffix}. Use JSON."
        raise ValueError(msg)

    with Path.open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        user_queries = [UserQuery(**item) for item in data]

    return user_queries


async def main() -> None:
    """Run verifications using the CoVe pipeline."""
    console = Console()
    parser = argparse.ArgumentParser(description="Run verification on text pairs")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file path (JSON) containing questions",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path to save batch results (JSON format)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="playground/critic/.env",
        help="Path to .env file (default: playground/critic/.env)",
    )
    parser.add_argument(
        "--yaml-file",
        type=str,
        default="playground/critic/config/config.yaml",
        help="Path to YAML config file (default: playground/critic/config/config.yaml)",
    )

    args = parser.parse_args()

    if args.input:
        console.print(f"[bold]Loading questions from[/] {args.input}")
        user_queries = load_questions(args.input)
        console.print(f"[green]Loaded[/] {len(user_queries)} questions")

        start = time.perf_counter()
        batch_result = await run_batch_verification(
            user_queries,
            env_file=args.env_file,
            yaml_file=args.yaml_file,
            console=console,
        )
        end = time.perf_counter()

        table = Table(title="Batch Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Time taken", f"{end - start:.2f}s")
        table.add_row("Total questions", str(batch_result.total_questions))
        table.add_row("Aligned answers", str(batch_result.aligned_count))
        table.add_row("Success rate", f"{batch_result.success_rate:.2%}")
        table.add_row("Average confidence", f"{batch_result.average_confidence:.2f}")
        console.print(table)

        if args.output:
            output_path = Path(args.output)
            output_json = json.dumps(batch_result.model_dump(), indent=4, ensure_ascii=False)
            await asyncio.to_thread(output_path.write_text, output_json, "utf-8")
            console.print(f"[green]Results saved to[/] {output_path}")
    else:
        console.rule("[bold cyan]CoVe Verification")
        await run_verification(
            user_query=UserQuery(
                question="Who was the first woman to win two Nobel Prizes in different scientific fields?"
            ),
            env_file=args.env_file,
            yaml_file=args.yaml_file,
            console=console,
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
uv run -m playground.critic.main \
    --env-file playground/critic/.env \
    --yaml-file playground/critic/config/config.yaml \
    --input playground/critic/questions.json \
    --output cove_results.json
"""
