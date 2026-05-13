# omniadapters (TypeScript)

A thin TypeScript adapter that maps a unified `ProviderConfig` into a [Vercel AI SDK](https://sdk.vercel.ai/) `LanguageModel`. Mirrors the philosophy of the Python `omniadapters/pydantic_ai/` module: don't reinvent the framework, just configure it.

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

## Playground

```bash
OPENAI_API_KEY=sk-... npx tsx playground/completion/01-demo.ts --provider openai
ANTHROPIC_API_KEY=sk-ant-... npx tsx playground/completion/01-demo.ts --provider anthropic --stream
```

CLI flags: `--provider {openai|anthropic|gemini|azure-openai|all}`, `--prompt <text>`, `--stream`, `--trace`.

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
