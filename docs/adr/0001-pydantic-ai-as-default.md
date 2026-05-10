# ADR 0001: pydantic-ai as the recommended default

Status: **Accepted** (2026-05-10)
Supersedes: —
Superseded by: —

## Context

`omniadapters` currently ships three adapter modules that wrap the same four
providers (OpenAI, Anthropic, Gemini, Azure OpenAI):

| Module        | Backbone     | What it provides                                          |
| ------------- | ------------ | --------------------------------------------------------- |
| `structify/`  | `instructor` | Structured outputs, partial streaming, 5-point hook trace |
| `completion/` | `instructor` | Non-structured chat completions, token streaming          |
| `pydantic_ai/`| `pydantic-ai`| Bridge from `ProviderConfig` to a `pydantic_ai.Agent`     |

Two of those are thick — they replicate everything the underlying SDKs do,
plus our own `BaseAdapter` lifecycle, hook system, mode plumbing, and result
wrappers.  The third is intentionally thin: pydantic-ai is itself an
opinionated framework, and wrapping it again would just duplicate the
abstraction it already provides.

Pydantic-ai 1.93 already covers, natively, every capability the legacy
modules provide:

| Capability             | Legacy surface (instructor)                  | Native pydantic-ai surface                                     |
| ---------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| Structured output      | `instructor.create(response_model=M)`        | `agent.run(prompt, output_type=M)`                             |
| Output strategy switch | `instructor.Mode.{TOOLS,JSON,…}`             | `NativeOutput`/`ToolOutput`/`PromptedOutput` wrappers + auto   |
| Streaming partials     | `instructor.create_partial(...)`             | `agent.run_stream(...)` → `run.stream_output()`                |
| Streaming raw text     | `completion.agenerate(stream=True)`          | `agent.run_stream(...)` → `run.stream_text(delta=True)`        |
| Multi-turn             | manual `messages.append(...)`               | `message_history=result.all_messages()`                        |
| Multimodal in          | `instructor.processing.multimodal.Image`     | `pydantic_ai.ImageUrl` / `BinaryContent`                       |
| Raw provider response  | typed generic on `BaseAdapter`               | `ModelResponse.vendor_details` (+ `vendor_id`, `vendor_metadata`) |
| Tracing                | five named hooks → `CompletionTrace`         | `event_stream_handler=` per call                                |
| Retries                | manual loop                                  | `ModelRetry` exception + `retries=`/`output_retries=`           |
| Telemetry              | (none built-in)                              | `instrument=` / Logfire integration                             |
| Tools / agents         | (out of scope for instructor)                | first-class `Tool`, `Toolset`, deps                             |

Maintaining two parallel stacks costs us:

- **Drift risk** — `instructor` and pydantic-ai both evolve; keeping the
  wrappers aligned with the SDKs and with each other is unbounded
  maintenance.
- **Cognitive load** — three modules, three mental models, three sets of
  tests, three sets of demos, for the same task.
- **Feature lag** — capabilities like Logfire instrumentation, deps-typed
  agents, and built-in tool routing exist only in the pydantic-ai branch.

## Decision

We make pydantic-ai the **recommended default** for new code.

`structify/` and `completion/` enter **maintenance mode**: bug fixes and
provider-bump compatibility, but no new features. The `pydantic_ai/`
module stays intentionally minimal — a `ProviderConfig → Model` bridge
plus a single small tracing utility — so callers learn pydantic-ai's real
API rather than ours.

We deliberately do **not**:

- Delete `structify/` or `completion/` in this change.
- Reshuffle `pyproject.toml` dependencies.
- Cut a major version.

Those moves are gated on this ADR being accepted in practice (see
"Future work").

## Migration mapping

| structify / completion                                            | pydantic-ai equivalent                                                       |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `structify.create_adapter(...).acreate(messages, M)`              | `pydantic_ai.create_adapter(...).create_agent(output_type=M).run(prompt)`    |
| `adapter.astream(messages, M)`                                    | `async with agent.run_stream(prompt) as run: async for p in run.stream_output(): ...` |
| `completion.create_adapter(...).agenerate(messages)`              | `agent.run(prompt)` (no `output_type` → `str`)                               |
| `completion.create_adapter(...).agenerate(messages, stream=True)` | `async with agent.run_stream(prompt) as run: async for c in run.stream_text(delta=True): ...` |
| `with_hooks=True` → `CompletionResult.trace`                      | `handler, trace = make_trace_recorder(); agent.run(..., event_stream_handler=handler)` |
| `result.trace.raw_response`                                       | `result.all_messages()` for `ModelResponse`s; raw SDK object via `ModelResponse.vendor_details` |
| `instructor.Mode` selection                                       | `output_type=NativeOutput(M)` / `ToolOutput(M)` / `PromptedOutput(M)` (or default auto) |
| `adapter.aclose()`                                                | `adapter.aclose()` (same shape; closes underlying provider HTTP client)      |
| Multi-turn via `messages.append(...)`                             | `result = await agent.run(p, message_history=result.all_messages())`         |
| `instructor.processing.multimodal.Image.from_url(u)`              | `ImageUrl(url=u)` (or `BinaryContent(data=..., media_type=...)`)             |

## Semantic deltas worth calling out

1. **"Raw response" semantics.** `structify`/`completion` return the bare
   provider SDK object via a typed generic. Pydantic-ai normalises that
   into `ModelResponse`; the original SDK payload lives on
   `ModelResponse.vendor_details`. Code that introspected provider-specific
   fields needs to read through `vendor_details`.

2. **Tracing is a stream, not five named hooks.** Instructor exposes
   `started`, `completed`, `parsed`, `error`, `last_attempt`. Pydantic-ai
   emits a discriminated event union (`PartStartEvent`, `PartDeltaEvent`,
   `FinalResultEvent`, `FunctionToolCallEvent`, …) through
   `event_stream_handler`. Our `make_trace_recorder` collects these into a
   plain list; richer instrumentation should use Logfire or the `instrument=`
   hook directly rather than re-deriving instructor's five points.

3. **`instructor.Mode` granularity is gone.** Pydantic-ai picks the
   strategy automatically from the model + schema; explicit selection is via
   the coarser `NativeOutput`/`ToolOutput`/`PromptedOutput` wrappers around
   `output_type`. The fine-grained instructor modes (`MD_JSON`,
   `JSON_SCHEMA`, `TOOLS_STRICT`, …) do not have 1:1 equivalents.

4. **No new opinionated adapter methods.** We deliberately did **not** add
   `arun`, `astream`, `astream_text`, or a `CompletionResult`-style wrapper
   to `PydanticAIAdapter`. Demos use `Agent.run` / `Agent.run_stream`
   directly so the user becomes proficient in pydantic-ai, not in our gloss
   over it.

## Capability matrix (post-migration)

Every legacy demo has a pydantic-ai counterpart in
`playground/pydantic_ai/`. Each row below points at the file that proves
the parity:

| Capability               | structify proof                                | pydantic_ai proof                                     |
| ------------------------ | ---------------------------------------------- | ----------------------------------------------------- |
| Structured one-shot      | `playground/structify/text/01_demo.py`         | `playground/pydantic_ai/text/01_demo.py`              |
| Multi-turn structured    | `playground/structify/text/02_conversation.py` | `playground/pydantic_ai/text/02_conversation.py`      |
| Streaming structured     | `--stream` flag on the structify demos         | `playground/pydantic_ai/text/03_streaming.py`         |
| Trace capture            | `--trace` flag on the structify demos          | `playground/pydantic_ai/text/04_traced.py`            |
| Multimodal structured    | `playground/structify/image/image_analysis.py` | `playground/pydantic_ai/image/image_analysis.py`      |
| Non-structured chat      | `playground/completion/01_demo.py`             | `playground/pydantic_ai/chat/01_basic_chat.py`        |
| Streaming text chat      | `playground/completion/01_demo.py --stream`    | `playground/pydantic_ai/chat/02_streaming_chat.py`    |
| Multimodal chat          | (none — implicit)                              | `playground/pydantic_ai/chat/03_multimodal_chat.py`   |

## Deliberately not ported (yet)

- **Critic / CoVe orchestrator.** `playground/critic/` is a four-stage
  agent flow tightly coupled to `structify`. Porting it cleanly to
  pydantic-ai is a non-trivial refactor (prompt rendering, agent-per-stage,
  shared deps) and is out of scope for this ADR. It is the obvious next
  port and a strong candidate to validate the deps-typed `Agent` surface.

- **Five-hook fidelity.** `make_trace_recorder` does not synthesise
  `started` / `parsed` / `last_attempt` from the event stream. If a
  downstream caller relies on those exact lifecycle points, they must
  either keep using `structify` or build a richer translator on top of the
  events list. We judged the cost of perfect fidelity higher than the
  benefit.

- **Fine-grained `instructor.Mode`.** As above, modes like `MD_JSON_SCHEMA`
  have no direct equivalent in pydantic-ai's coarser strategy wrappers.
  Code that depends on a specific mode keeps using `structify` until
  pydantic-ai exposes equivalent control.

## Future work

When we are confident this ADR holds in practice (i.e. the pydantic-ai
demos cover real downstream usage):

1. Port the critic orchestrator (its own ADR / PR).
2. Delete `omniadapters/structify/` and `omniadapters/completion/`.
3. Drop `instructor` from the runtime dependencies in `pyproject.toml`.
4. Cut a major version with a migration changelog pointing at this ADR.

Steps (2)–(4) are explicitly **not** part of the change that introduced
this ADR.
