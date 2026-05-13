/**
 * Completion Adapter Demo — Multi-Provider LLM Completions over Vercel AI SDK.
 *
 * Mirrors playground/completion/01_demo.py.
 *
 * Usage:
 *   npx tsx playground/completion/01-demo.ts --provider all
 *   npx tsx playground/completion/01-demo.ts --provider openai
 *   npx tsx playground/completion/01-demo.ts --provider anthropic
 *   npx tsx playground/completion/01-demo.ts --provider gemini
 *   npx tsx playground/completion/01-demo.ts --provider azure-openai
 *   npx tsx playground/completion/01-demo.ts --prompt "Explain quantum computing"
 *   npx tsx playground/completion/01-demo.ts --stream
 *   npx tsx playground/completion/01-demo.ts --provider openai --stream --trace
 *
 * Env (loaded from process env; populate via .env or shell):
 *   OPENAI_API_KEY
 *   ANTHROPIC_API_KEY
 *   GOOGLE_GENERATIVE_AI_API_KEY   (or GEMINI_API_KEY)
 *   AZURE_OPENAI_API_KEY, AZURE_OPENAI_RESOURCE_NAME
 */

import { parseArgs } from "node:util";

import { createAdapter, type Provider } from "../../src/index";

interface Args {
  provider: Provider | "all";
  prompt: string;
  stream: boolean;
  trace: boolean;
}

function parseCliArgs(): Args {
  const { values } = parseArgs({
    options: {
      provider: { type: "string", default: "all" },
      prompt: { type: "string", default: "Explain recursion in programming in 2 sentences." },
      stream: { type: "boolean", default: false },
      trace: { type: "boolean", default: false },
    },
  });
  return {
    provider: values.provider as Args["provider"],
    prompt: values.prompt as string,
    stream: Boolean(values.stream),
    trace: Boolean(values.trace),
  };
}

interface ProviderDefaults {
  modelName: string;
  envKey: string;
  build: (apiKey: string) => Parameters<typeof createAdapter>[0]["providerConfig"];
}

const DEFAULTS: Record<Provider, ProviderDefaults> = {
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

function getDefaults(provider: Provider): ProviderDefaults {
  const defaults = DEFAULTS[provider];
  if (!defaults) throw new Error(`No defaults registered for provider: ${provider}`);
  return defaults;
}

function readApiKey(provider: Provider): string | null {
  const envKey = getDefaults(provider).envKey;
  const fallback = provider === "gemini" ? process.env.GEMINI_API_KEY : undefined;
  return process.env[envKey] ?? fallback ?? null;
}

const SYSTEM = "Always start your response with 'Hello, world!'";

async function runOnce(provider: Provider, args: Args): Promise<void> {
  const apiKey = readApiKey(provider);
  const defaults = getDefaults(provider);
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
    const result = await adapter.stream({
      system: SYSTEM,
      prompt: args.prompt,
    });
    let chunkCount = 0;
    for await (const chunk of result.textStream) {
      process.stdout.write(chunk);
      chunkCount += 1;
    }
    const finalUsage = await result.usage;
    console.log(`\n\n[chunks=${chunkCount}, usage=${JSON.stringify(finalUsage)}]`);
  } else {
    const result = await adapter.generate({
      system: SYSTEM,
      prompt: args.prompt,
    });
    console.log(result.text);
    console.log(`\n[model=${defaults.modelName}, usage=${JSON.stringify(result.usage)}]`);
    if (args.trace) {
      console.log("\n--- trace ---");
      console.log(JSON.stringify({ finishReason: result.finishReason, usage: result.usage }, null, 2));
    }
  }
}

async function main(): Promise<void> {
  const args = parseCliArgs();
  const providers: Provider[] =
    args.provider === "all"
      ? (["openai", "anthropic", "gemini", "azure-openai"] as const).slice()
      : [args.provider];

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
