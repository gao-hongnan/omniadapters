# Chain of verification, stage by stage

Chain-of-Verification (CoVe) reduces hallucination in four steps: draft an answer, plan
questions that would expose its errors, answer those questions independently, then
write a final answer from the evidence
([Dhuliawala et al. 2023](https://arxiv.org/abs/2309.11495)).
`omniadapters.services.cove` stages those steps as a courtroom. Each step is a _role_,
and each role receives a _brief_ that decides what it may see.

This page follows one question through the factual-QA recipe,
`omniadapters.services.cove.recipes.factual_qa`. The prompts are the recipe's real
prompts, rendered for this question. The replies are illustrative: they show what a
careful model would return, not output captured from a live model.
[The dataflow diagram](diagrams/cove-dataflow.drawio) shows the same run on one page;
open it in draw.io.

## The cast

| Paper step                     | Role                      | Sees (its brief)                                | Returns                        |
| ------------------------------ | ------------------------- | ----------------------------------------------- | ------------------------------ |
| 1. Draft a baseline answer     | Proponent                 | `CaseBrief(case)`                               | the proposal, an `Answer`      |
| 2. Plan verification questions | Contrarian                | `ContrarianBrief(case, proposal)`               | the challenges, `Challenges`   |
| 3. Answer them independently   | Witness, one per question | `WitnessBrief(case, challenge)`                 | testimony, a `WitnessAnswer`   |
| 4. Write the verified answer   | Judge                     | `JudgeBrief(case, proposal, evidence, hearing)` | the ruling, a `VerifiedAnswer` |

The other procedures add three more roles, the panel, the inquisitor and the
cross-examiner. They appear under "Other procedures" below.

## The question

> Who was the first woman to win two Nobel Prizes, and in which years?

The right answer is Marie Curie: Physics in 1903, Chemistry in 1911. In this run the
proponent gets the Chemistry year wrong and writes 1908. Follow the 1908 through each
stage to see where it is caught.

## Stage 1: the proponent drafts an answer

The proponent sees a `CaseBrief`, which holds the case and nothing else.

```text
[system]
You are a knowledgeable assistant. Answer the question directly and completely. State the specific facts your answer relies on: names, dates, numbers, places.

[user]
Who was the first woman to win two Nobel Prizes, and in which years?
```

It returns the proposal:

```python
Answer(text="Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908.")
```

The proposal is only the answer. It carries no confidence score or self-assessment, so
nothing in it can anchor the roles that follow.

## Stage 2: the contrarian plans verification questions

The contrarian sees a `ContrarianBrief`: the case and the proposal. Its job is to doubt
the proposal. It finds each fact the answer relies on and turns it into a question that
can be answered _without_ reading the proposal.

```text
[system]
You are a contrarian fact-checker. The proposed answer may contain factual errors. Write 3 to 6 open verification questions, one per factual claim the answer relies on: names, dates, numbers, places, relationships. Each question must be answerable on its own, without the proposed answer, and must ask for the fact instead of stating it: ask "When did X happen?", never "Did X happen in 1908?".

[user]
Question: Who was the first woman to win two Nobel Prizes, and in which years?
Proposed answer: Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908.
```

It returns the challenges:

```python
Challenges(
    questions=[
        "In which year did Marie Curie win the Nobel Prize in Physics?",
        "In which year did Marie Curie win the Nobel Prize in Chemistry?",
        "Who was the first woman to win two Nobel Prizes?",
    ]
)
```

Three things keep the questions useful:

- **Open, not yes/no.** Each question asks _for_ a fact ("In which year…?") and never
  asks to confirm one ("Did she win in 1908?"). Models tend to agree with a yes/no
  question, which is exactly how the draft's error would slip through.
- **One owner for the bounds.** `MIN_CHALLENGES = 3` and `MAX_CHALLENGES = 6` feed both
  the `Challenges` schema and the prompt, so the model is told the same limits it is
  validated against.
- **An engine-level cost guard.** The step's `parse` turns `Challenges` into a plain
  `list[str]`. The court then admits the list only if its length is within
  `Limits.min_challenges` to `Limits.max_challenges`, 1 to 12 by default, and raises
  `ChallengeCountError` otherwise. That bounds the fan-out that follows.

## Stage 3: the witnesses answer, each on their own

Each question goes to its own witness. Every witness sees a `WitnessBrief`: the case and
one challenge. **There is no proposal field.** A witness cannot be swayed by the draft,
because the draft is never handed to it. This is the paper's central finding: answers
that can see the draft repeat its mistakes. The brief's type enforces it, and building a
recipe fails if a witness template mentions `proposal`.

The witness for the second question is shown only the question. The factual-QA witness
prompt does not use the case at all:

```text
[system]
You are a careful researcher. Answer the question accurately and concisely from your own knowledge. If you are not sure, say so instead of guessing.

[user]
In which year did Marie Curie win the Nobel Prize in Chemistry?
```

The three witnesses testify:

| # | Challenge                                                       | Testimony                                                                             |
| - | --------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1 | In which year did Marie Curie win the Nobel Prize in Physics?   | `1903`: she shared it with Pierre Curie and Henri Becquerel for work on radioactivity |
| 2 | In which year did Marie Curie win the Nobel Prize in Chemistry? | `1911`: she won it alone, for discovering the elements polonium and radium            |
| 3 | Who was the first woman to win two Nobel Prizes?                | `Marie Curie`: she won the 1903 Physics prize and the 1911 Chemistry prize            |

The second witness, never having seen the draft, gives 1911.

The witnesses run concurrently, each on its own `Stand` with a `witness_timeout_s`
deadline, 60 seconds by default. A witness that raises or times out does not end the
hearing. Its challenge becomes an `Unanswered` outcome, and the quorum,
`Limits.quorum`, decides whether enough challenges were answered. The default quorum is
1.0, which ends the hearing with `QuorumNotMetError` on the first unanswered challenge.
The outcomes form the `Evidence`, in the order the contrarian asked.

## Stage 4: the judge rules

The judge sees a `JudgeBrief`: the case, the proposal, the evidence and the hearing
number. It is the only role that weighs the proposal against independent testimony.

```text
[system]
You are the judge in a chain of verification. You are given a question, a proposed answer, and evidence from independent witnesses who never saw the proposed answer. Where the evidence contradicts the proposal, write a corrected answer from the evidence; where it supports the proposal, confirm it. Give the verdict, a confidence between 0 and 1, and the evidence that decided it.

[user]
Question: Who was the first woman to win two Nobel Prizes, and in which years?
Proposed answer: Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908.

Evidence:
- Q: In which year did Marie Curie win the Nobel Prize in Physics?
  A: 1903 (She shared it with Pierre Curie and Henri Becquerel for work on radioactivity.)
- Q: In which year did Marie Curie win the Nobel Prize in Chemistry?
  A: 1911 (She won it alone, for discovering the elements polonium and radium.)
- Q: Who was the first woman to win two Nobel Prizes?
  A: Marie Curie (She won the 1903 Physics prize and the 1911 Chemistry prize.)
```

It returns the ruling:

```python
VerifiedAnswer(
    answer="Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
    verdict=Verdict.REVISED,
    confidence=0.9,
    rationale="An independent witness dates the Chemistry prize to 1911, not 1908; the other answers support the proposal.",
)
```

The ruling carries the corrected answer, not just a verdict. That is the paper's final
step: rewrite the answer from the evidence.

## Stage 5: a retrial, if you configure one

The factual-QA recipe holds a single hearing, so this run ends with the disposition
`single_hearing`. A recipe can opt into retrials:

```python
ChainOfVerification(
    proponent=...,
    procedure=...,
    judge=...,
    retrial=Retrial(max_hearings=3, reconsider=reconsider),
)
```

After each ruling the engine calls `reconsider(ruling)`.

- Returning `Reopen(proposal=...)` starts another hearing on the revised proposal,
  beginning again at the contrarian. The proponent is not called again.
- Returning `None` lets the ruling stand, and the disposition is `settled`.
- If `reconsider` still wants another hearing when `max_hearings` is reached, the
  disposition is `hearing_limit`. Treat that ruling as unsettled, for example by sending
  it to a human.

The invoice recipe in `tests/unit/services/cove/test_deterministic_witness.py` uses a
retrial to re-extract a misread total and verify it again.

## The transcript

`run` returns a `Transcript`, the complete record of the run. `transcript.ruling` is the
last hearing's ruling, and `transcript.to_json()` serialises everything. Here is this
run, with the run id shortened and step timings left out:

```json
{
  "run_id": "3f9c…",
  "case": {"text": "Who was the first woman to win two Nobel Prizes, and in which years?"},
  "hearings": [
    {
      "number": 1,
      "proposal": {"text": "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908."},
      "evidence": {
        "outcomes": [
          {
            "challenge": "In which year did Marie Curie win the Nobel Prize in Physics?",
            "testimony": {"answer": "1903", "explanation": "She shared it with Pierre Curie and Henri Becquerel for work on radioactivity."},
            "kind": "answered"
          },
          {
            "challenge": "In which year did Marie Curie win the Nobel Prize in Chemistry?",
            "testimony": {"answer": "1911", "explanation": "She won it alone, for discovering the elements polonium and radium."},
            "kind": "answered"
          },
          {
            "challenge": "Who was the first woman to win two Nobel Prizes?",
            "testimony": {"answer": "Marie Curie", "explanation": "She won the 1903 Physics prize and the 1911 Chemistry prize."},
            "kind": "answered"
          }
        ]
      },
      "ruling": {
        "answer": "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
        "verdict": "revised",
        "confidence": 0.9,
        "rationale": "An independent witness dates the Chemistry prize to 1911, not 1908; the other answers support the proposal."
      }
    }
  ],
  "disposition": "single_hearing",
  "record": [
    {"role": "proponent", "hearing": 1, "challenge_index": null, "kind": "step_completed"},
    {"role": "contrarian", "hearing": 1, "challenge_index": null, "kind": "step_completed"},
    {"role": "witness", "hearing": 1, "challenge_index": 0, "kind": "step_completed"},
    {"role": "witness", "hearing": 1, "challenge_index": 1, "kind": "step_completed"},
    {"role": "witness", "hearing": 1, "challenge_index": 2, "kind": "step_completed"},
    {"role": "judge", "hearing": 1, "challenge_index": null, "kind": "step_completed"}
  ]
}
```

The `record` lists every step that finished or failed, in the order they ended. Observers
receive those events too, plus each step's start and the run's start and finish, while
the run is in progress. That is how the critic CLI prints its live step log.

## Other procedures

The procedure decides how a hearing gathers evidence. The four variants come from the
paper, and no single one wins on every task.

| Procedure            | Calls per hearing                              | Who sees the proposal       | On this question          |
| -------------------- | ---------------------------------------------- | --------------------------- | ------------------------- |
| `Factored` (default) | contrarian, then one witness per question      | contrarian                  | catches 1908              |
| `TwoStep`            | contrarian, then one panel                     | contrarian                  | catches 1908              |
| `Joint`              | one inquisitor                                 | inquisitor                  | tends to repeat 1908      |
| `FactorRevise`       | factored, plus one cross-examiner per question | contrarian, cross-examiners | catches 1908 and flags it |

Choose `factored`, `two_step` or `joint` with `procedure:` in the recipe's YAML. Build
the factor+revise variant in code with `build_cross_examined_factual_qa`.

### TwoStep: one panel answers every question

The panel sees a `PanelBrief`: the case and every challenge, still without the proposal.
It answers all three questions in one call and returns a `PanelAnswers`. That is cheaper
than one witness per question, but the panel must return exactly one answer per question,
or the hearing fails with `PanelMismatchError`.

### Joint: one inquisitor asks and answers

The inquisitor sees a `ContrarianBrief`, so it reads the proposal, and it asks and
answers its questions in a single call, returning an `Inquiry`. It is the cheapest
variant and the weakest. An inquisitor that has just read "Chemistry in 1908" tends to
answer the Chemistry question with 1908. The judge then sees evidence that agrees with
the draft and confirms the wrong answer.

### FactorRevise: a cross-examiner checks each testimony

This is factored testimony plus one cross-examiner per question. The cross-examiner sees
a `CrossExaminationBrief`: the proposal, one question, and that question's independent
testimony. Its only job is to say whether the two agree. For the second question:

```text
[system]
You are a cross-examiner. Decide whether an independently obtained answer agrees with the proposed answer on the point the verification question asks about.

[user]
Proposed answer: Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1908.
Verification question: In which year did Marie Curie win the Nobel Prize in Chemistry?
Independent answer: 1911 (She won it alone, for discovering the elements polonium and radium.)
```

```python
Consistency(consistent=False, note="The independent answer gives 1911; the proposal says 1908.")
```

The judge then reads an explicit contradiction instead of having to spot one. The paper
found this variant strongest on long-form answers, at the cost of one extra call per
question.

## Inside one step: the contrarian on pydantic-ai

Every role above is one call to an `llm_step`. When the roles come from config, that call
runs on a pydantic-ai agent. Here is the contrarian's call from start to finish.

1. The `JinjaPrompt` renders the `ContrarianBrief` into `Messages(system=..., user=...)`,
   the two texts shown in stage 2.
1. `llm_step` calls `acreate(messages, Challenges)` on the role's `StructuredLLM`.
   Optional decorators can wrap that call: `Throttled` caps concurrent calls, and
   `Budgeted` spends from a shared `CallBudget`.
1. `PydanticAIStructuredLLM` runs the agent once, with the user message as the prompt,
   the system message as the run's instructions, and `Challenges` as the output type.
   With `output_mode: auto`, pydantic-ai asks the model's profile how to get structured
   output. For most models, OpenAI's included, the answer is a tool call, so the model
   is offered one output tool, `final_result`. Its description is the response model's
   docstring, and its parameters are the response model's JSON schema:

    ```json
    {
      "additionalProperties": false,
      "properties": {
        "questions": {
          "description": "Open questions, one per factual claim, each answerable without the proposed answer.",
          "items": {"type": "string"},
          "maxItems": 6,
          "minItems": 3,
          "type": "array"
        }
      },
      "required": ["questions"],
      "title": "Challenges",
      "type": "object"
    }
    ```

1. Suppose the model calls `final_result` with only two questions. pydantic-ai validates
   the arguments against `Challenges`, finds the list too short, and sends the errors
   back to the model:

    ````text
    1 validation error:
    ```json
    [
      {
        "type": "too_short",
        "loc": [
          "questions"
        ],
        "msg": "List should have at least 3 items after validation, not 2",
        "input": [
          "In which year did Marie Curie win the Nobel Prize in Physics?",
          "In which year did Marie Curie win the Nobel Prize in Chemistry?"
        ]
      }
    ]
    ```

    Fix the errors and try again.
    ````

1. The model tries again with three questions. That reply validates, `acreate` returns a
   `Challenges` instance, and `parse` turns it into the `list[str]` the court admits.

If no reply validates within the role's `output_retries`, 1 by default, the step raises
pydantic-ai's `UnexpectedModelBehavior`. A failed contrarian fails the run with
`StepFailedError`; a failed witness becomes an `Unanswered` challenge.

The other output modes change only how the schema reaches the model. `output_mode:
native` uses the provider's own structured output instead of a tool, and `output_mode:
prompted` puts the schema in the instructions and parses JSON out of the reply text.

## Run it yourself

With API keys in `playground/critic/.env`, the critic CLI runs this recipe against real
models:

```bash
uv run -m playground.critic.main \
    --env-file playground/critic/.env \
    --yaml-file playground/critic/config/config.yaml
```

From code, given a validated `FactualQAConfig`:

```python
from omniadapters.services.cove.recipes.factual_qa import FactualQAConfig, Question, open_factual_qa


async def verify(config: FactualQAConfig, question: str) -> None:
    async with open_factual_qa(config) as cove:
        transcript = await cove.run(Question(text=question))
    print(transcript.disposition, transcript.ruling.verdict, transcript.ruling.answer)
```

For tests, `omniadapters.services.cove.testing.ScriptedLLM` plays any role offline and
records exactly what each role was shown.
