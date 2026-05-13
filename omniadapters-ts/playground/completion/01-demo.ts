/**
 * Completion Adapter Demo — Multi-Provider LLM Completions over Vercel AI SDK.
 *
 * Mirrors playground/completion/01_demo.py.
 *
 *   npx tsx playground/completion/01-demo.ts --provider all
 *   npx tsx playground/completion/01-demo.ts --provider openai
 *   npx tsx playground/completion/01-demo.ts --provider anthropic --stream
 *   npx tsx playground/completion/01-demo.ts --provider gemini --prompt "What is AI?"
 *   npx tsx playground/completion/01-demo.ts --provider azure-openai --trace
 *
 * Env:
 *   OPENAI_API_KEY
 *   ANTHROPIC_API_KEY
 *   GOOGLE_GENERATIVE_AI_API_KEY   (or GEMINI_API_KEY)
 *   AZURE_OPENAI_API_KEY, AZURE_OPENAI_RESOURCE_NAME
 */

import { parseArgs } from "node:util";

import { z } from "zod";

import {
  PROVIDERS,
  type Provider,
  type ProviderConfig,
  createAdapter,
} from "../../src/index";

const CliArgsSchema = z.object({
  provider: z.union([z.enum(PROVIDERS), z.literal("all")]).default("all"),
  prompt: z.string().min(1).default("Explain recursion in programming in 2 sentences."),
  stream: z.boolean().default(false),
  trace: z.boolean().default(false),
});

type CliArgs = z.infer<typeof CliArgsSchema>;

function parseCliArgs(): CliArgs {
  const { values } = parseArgs({
    options: {
      provider: { type: "string", default: "all" },
      prompt: { type: "string" },
      stream: { type: "boolean", default: false },
      trace: { type: "boolean", default: false },
    },
  });
  return CliArgsSchema.parse(values);
}

interface ProviderDefaults {
  modelName: string;
  envKey: string;
  build: (apiKey: string) => ProviderConfig;
}

const DEFAULTS: { [P in Provider]: ProviderDefaults } = {
  openai: {
    modelName: "gpt-4o-mini",
    envKey: "OPENAI_API_KEY",
    build: (apiKey) => ({ provider: "openai", apiKey }),
  },
  anthropic: {
    modelName: "claude-3-5-haiku-20241022",
    envKey: "ANTHROPIC_API_KEY",
    build: (apiKey) => ({ provider: "anthropic", apiKey }),
  },
  gemini: {
    modelName: "gemini-2.0-flash-exp",
    envKey: "GOOGLE_GENERATIVE_AI_API_KEY",
    build: (apiKey) => ({ provider: "gemini", apiKey }),
  },
  "azure-openai": {
    modelName: "gpt-4o-mini",
    envKey: "AZURE_OPENAI_API_KEY",
    build: (apiKey) => ({
      provider: "azure-openai",
      apiKey,
      resourceName: process.env.AZURE_OPENAI_RESOURCE_NAME ?? "",
    }),
  },
};

function readApiKey(provider: Provider): string | null {
  const defaults = DEFAULTS[provider];
  const fallback = provider === "gemini" ? process.env.GEMINI_API_KEY : undefined;
  return process.env[defaults.envKey] ?? fallback ?? null;
}

const SYSTEM = "Always start your response with 'Hello, world!'";

async function runOnce(provider: Provider, args: CliArgs): Promise<void> {
  const defaults = DEFAULTS[provider];
  const apiKey = readApiKey(provider);
  if (!apiKey) {
    console.error(`[${provider}] missing ${defaults.envKey}, skipping`);
    return;
  }
  if (provider === "azure-openai" && !process.env.AZURE_OPENAI_RESOURCE_NAME) {
    console.error(`[${provider}] missing AZURE_OPENAI_RESOURCE_NAME, skipping`);
    return;
  }

  const adapter = createAdapter({
    providerConfig: defaults.build(apiKey),
    modelName: defaults.modelName,
  });

  console.log(`\n=== ${provider} (${defaults.modelName}) ===`);

  if (args.stream) {
    const result = await adapter.stream({ system: SYSTEM, prompt: args.prompt });
    let chunks = 0;
    for await (const chunk of result.textStream) {
      process.stdout.write(chunk);
      chunks += 1;
    }
    console.log(`\n\n[chunks=${chunks}, usage=${JSON.stringify(await result.usage)}]`);
    return;
  }

  const result = await adapter.generate({ system: SYSTEM, prompt: args.prompt });
  console.log(result.text);
  console.log(`\n[model=${defaults.modelName}, usage=${JSON.stringify(result.usage)}]`);
  if (args.trace) {
    console.log("--- trace ---");
    console.log(JSON.stringify({ finishReason: result.finishReason, usage: result.usage }, null, 2));
  }
}

async function main(): Promise<void> {
  const args = parseCliArgs();
  const providers: Provider[] = args.provider === "all" ? [...PROVIDERS] : [args.provider];

  console.log(`Prompt: ${args.prompt}`);
  for (const provider of providers) {
    try {
      await runOnce(provider, args);
    } catch (err) {
      console.error(`[${provider}] error:`, err instanceof Error ? err.message : err);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
