# omniadapters (TypeScript)

A thin TypeScript adapter that maps a unified `ProviderConfig` into a [Vercel AI SDK](https://sdk.vercel.ai/) `LanguageModel`. Mirrors the philosophy of the Python `omniadapters/pydantic_ai/` module: don't reinvent the framework, just configure it.

**Targets**: `ai@^6`, `@ai-sdk/*@^3`, `zod@^4`, Node `>=20`.

**Toolchain**: pnpm (package manager) · oxlint (Rust-based lint, the ruff-equivalent) · tsc (type-check) · vitest (test) · tsup (build).

Supports four providers via the official `@ai-sdk/*` packages:

| Provider       | Underlying package    |
| -------------- | --------------------- |
| `openai`       | `@ai-sdk/openai`      |
| `anthropic`    | `@ai-sdk/anthropic`   |
| `gemini`       | `@ai-sdk/google`      |
| `azure-openai` | `@ai-sdk/azure`       |

The provider discriminator and config shapes match the Python `ProviderConfig` (`omniadapters/core/models.py`) so code written against one ecosystem reads naturally in the other.

## Install

```bash
npm install omniadapters ai zod
# install only the provider SDKs you need:
npm install @ai-sdk/openai @ai-sdk/anthropic @ai-sdk/google @ai-sdk/azure
```

All `@ai-sdk/*` packages are declared as **optional** peer dependencies, so a missing one only errors when you actually try to construct that provider's model.

## Usage

### Non-streaming completion

```ts
import { createAdapter } from "omniadapters";

const adapter = createAdapter({
  providerConfig: { provider: "openai", apiKey: process.env.OPENAI_API_KEY! },
  modelName: "gpt-4o-mini",
});

const result = await adapter.generate({
  system: "Be concise.",
  prompt: "Explain recursion in one sentence.",
});

console.log(result.text);
console.log(result.usage);
```

### Streaming

```ts
const stream = await adapter.stream({
  prompt: "Write a haiku about TypeScript.",
});

for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
console.log("\nusage:", await stream.usage);
```

### Get the raw `LanguageModel` and use any AI SDK function

`adapter.model()` returns a Vercel AI SDK `LanguageModel`, so anything in the AI SDK works:

```ts
import { generateObject, embed } from "ai";
import { z } from "zod";

const model = await adapter.model();

const { object } = await generateObject({
  model,
  schema: z.object({ city: z.string(), country: z.string() }),
  prompt: "Pick a random capital city.",
});

// Embeddings, tool calls, image generation, etc. all work the same way.
```

### Provider config shapes

```ts
// OpenAI
{ provider: "openai", apiKey: "sk-...", organization?: "org-...", project?: "proj-...", baseURL?: "..." }

// Anthropic
{ provider: "anthropic", apiKey: "sk-ant-...", baseURL?: "..." }

// Gemini
{ provider: "gemini", apiKey: "AIza...", baseURL?: "..." }

// Azure OpenAI
{ provider: "azure-openai", apiKey: "...", resourceName: "my-resource", apiVersion?: "...", baseURL?: "..." }
```

Validation is enforced by `zod` at `createAdapter` call time. Pass `validate: false` to skip.

## Development

```bash
# pnpm is the package manager (corepack-managed via the `packageManager` field).
# If you don't have pnpm: `corepack enable` or `npm i -g pnpm`.
cd omniadapters-ts
pnpm install --frozen-lockfile

pnpm run lint        # oxlint
pnpm run typecheck   # tsc --noEmit
pnpm run test        # vitest
pnpm run build       # tsup (ESM + CJS + .d.ts)
pnpm run ci          # all four, sequenced
```

## Playground

```bash
OPENAI_API_KEY=sk-... npx tsx playground/completion/01-demo.ts --provider openai
ANTHROPIC_API_KEY=sk-ant-... npx tsx playground/completion/01-demo.ts --provider anthropic --stream
```

CLI flags: `--provider {openai|anthropic|gemini|azure-openai|all}`, `--prompt <text>`, `--stream`, `--trace`.

## Migrating from AI SDK v4

If you're porting code that targeted `ai@4`, note these renames in the AI SDK itself (your `adapter.generate(...)` and `adapter.stream(...)` options forward straight to the SDK, so the new names apply):

| AI SDK v4 | AI SDK v5/v6 |
| --- | --- |
| `CoreMessage` | `ModelMessage` |
| `maxTokens` | `maxOutputTokens` |
| `usage.promptTokens` | `usage.inputTokens` |
| `usage.completionTokens` | `usage.outputTokens` |

`LanguageModel` is still exported (now an alias over `LanguageModelV2`/`V3`), and `generateText` / `streamText` / `createOpenAI` / `createAnthropic` / `createGoogleGenerativeAI` / `createAzure` keep their call shapes.

## Why a single adapter (not one per provider)?

The Python `completion/adapters/*.py` adapters do real work: normalizing each provider's response/streaming shape into a unified `CompletionResponse` / `StreamChunk` / `Usage`. The Vercel AI SDK already does that normalization — `generateText` and `streamText` return the same shape regardless of provider. A per-provider TS adapter would be pure forwarding with nothing to normalize. Same reason the Python `pydantic_ai/adapter.py` is one class, not four.

## Scope

- Completion (`generate`) and streaming (`stream`) wired as convenience methods.
- `generateObject`, `streamObject`, tool calling, embeddings, and image models all work by passing the result of `adapter.model()` to the appropriate AI SDK function — no additional adapter code required.

## Layout

```
src/
  core/
    config.ts    # zod ProviderConfig (discriminated union)
    errors.ts    # missing-package error messages
  adapter.ts     # VercelAIAdapter class
  factory.ts     # createAdapter() with per-provider overloads
  index.ts       # public exports
playground/
  completion/
    01-demo.ts   # CLI demo (port of playground/completion/01_demo.py)
tests/
  adapter.test.ts
```

## License

Apache-2.0 (matches the parent project).
