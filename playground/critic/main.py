r"""Verify answers to factual questions with chain of verification: the factual-QA recipe, from YAML config.

    uv run -m playground.critic.main \
        --env-file playground/critic/.env \
        --yaml-file playground/critic/config/config.yaml \
        --input playground/critic/questions.json \
        --output cove_results.json

Without ``--input`` it verifies one built-in question. Each role's API key comes from the env file
(``COVE__<ROLE>__PROVIDER_CONFIG__API_KEY``); everything else comes from the YAML file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pydantic_core
from pydantic import TypeAdapter
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from omniadapters.services.cove import CoVeRunError
from omniadapters.services.cove.events import StepCompleted, StepRaised
from omniadapters.services.cove.recipes.factual_qa import Question, open_factual_qa

from .config.settings import load_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omniadapters.services.cove import CoVeEvent, Transcript
    from omniadapters.services.cove.recipes.factual_qa import Answer, VerifiedAnswer, WitnessAnswer

type FactualQATranscript = Transcript[Question, Answer, str, WitnessAnswer, VerifiedAnswer]

_DEFAULT_QUESTION: Final = "Who was the first woman to win two Nobel Prizes in different scientific fields?"
_QUESTIONS: Final = TypeAdapter(list[Question])


@dataclass(frozen=True, slots=True)
class Arguments:
    questions_file: Path | None
    output_file: Path | None
    env_file: Path
    yaml_file: Path


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="Verify factual answers with chain of verification.")
    parser.add_argument("--input", "-i", dest="questions_file", type=Path, help='JSON list of {"text": ...}')
    parser.add_argument("--output", "-o", dest="output_file", type=Path, help="write every transcript here")
    parser.add_argument("--env-file", type=Path, default=Path("playground/critic/.env"), help="API keys")
    parser.add_argument(
        "--yaml-file", type=Path, default=Path("playground/critic/config/config.yaml"), help="recipe config"
    )
    namespace = parser.parse_args()
    return Arguments(
        questions_file=namespace.questions_file,
        output_file=namespace.output_file,
        env_file=namespace.env_file,
        yaml_file=namespace.yaml_file,
    )


class ConsoleObserver:
    """Log every role's step as the court works through the docket."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def __call__(self, event: CoVeEvent) -> None:
        match event:
            case StepCompleted():
                self._console.log(
                    f"[cyan]{event.run_id[:8]}[/] hearing {event.hearing} "
                    f"{event.role}{_position(event.challenge_index)} done in {event.elapsed_s:.2f}s"
                )
            case StepRaised():
                self._console.log(
                    f"[red]{event.run_id[:8]}[/] hearing {event.hearing} "
                    f"{event.role}{_position(event.challenge_index)} raised {event.error_type}: {escape(event.message)}"
                )
            case _:
                return


def _position(challenge_index: int | None) -> str:
    return "" if challenge_index is None else f"[{challenge_index}]"


def load_questions(path: Path) -> list[Question]:
    return _QUESTIONS.validate_json(path.read_bytes())


def render(
    console: Console, questions: Sequence[Question], results: Sequence[FactualQATranscript | CoVeRunError]
) -> None:
    table = Table(title="Chain of verification")
    table.add_column("Question", overflow="fold")
    table.add_column("Verdict")
    table.add_column("Confidence", justify="right")
    table.add_column("Final answer", overflow="fold")
    for question, result in zip(questions, results, strict=True):
        if isinstance(result, CoVeRunError):
            table.add_row(escape(question.text), "[red]failed[/]", "-", escape(str(result)))
        else:
            ruling = result.ruling
            table.add_row(
                escape(question.text), ruling.verdict.value, f"{ruling.confidence:.2f}", escape(ruling.answer)
            )
    console.print(table)


def report(
    questions: Sequence[Question], results: Sequence[FactualQATranscript | CoVeRunError]
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for question, result in zip(questions, results, strict=True):
        if isinstance(result, CoVeRunError):
            entries.append(
                {"question": question.text, "error": {"type": type(result).__name__, "message": str(result)}}
            )
        else:
            entries.append({"question": question.text, "transcript": pydantic_core.to_jsonable_python(result)})
    return entries


async def main() -> None:
    arguments = parse_arguments()
    console = Console()
    settings = load_settings(yaml_file=arguments.yaml_file, env_file=arguments.env_file)
    questions = (
        load_questions(arguments.questions_file) if arguments.questions_file else [Question(text=_DEFAULT_QUESTION)]
    )

    console.rule(f"[bold cyan]Chain of verification: {len(questions)} question(s), {settings.cove.procedure.value}")
    async with open_factual_qa(settings.cove, observers=[ConsoleObserver(console)]) as cove:
        results = await cove.run_many(questions)
    render(console, questions, results)

    if arguments.output_file is not None:
        document = json.dumps(report(questions, results), indent=2, ensure_ascii=False)
        await asyncio.to_thread(arguments.output_file.write_text, document, "utf-8")
        console.print(f"[green]Transcripts saved to[/] {arguments.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
