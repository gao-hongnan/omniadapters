import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  AnthropicProviderConfigSchema,
  AzureOpenAIProviderConfigSchema,
  GeminiProviderConfigSchema,
  OpenAIProviderConfigSchema,
  PROVIDERS,
  PROVIDER_SCHEMAS,
  ProviderConfigSchema,
  VercelAIAdapter,
  createAdapter,
  isProviderConfig,
  parseProviderConfig,
  safeParseProviderConfig,
} from "../src/index";

type ModelFactory = (id: string) => unknown;
type ProviderFactory = (settings: Record<string, unknown>) => ModelFactory;

function mockProviderFactory(name: string): {
  sentinel: symbol;
  modelFactory: ReturnType<typeof vi.fn<ModelFactory>>;
  providerFactory: ReturnType<typeof vi.fn<ProviderFactory>>;
} {
  const sentinel = Symbol(name);
  const modelFactory = vi.fn<ModelFactory>(() => sentinel);
  const providerFactory = vi.fn<ProviderFactory>(() => modelFactory);
  return { sentinel, modelFactory, providerFactory };
}

beforeEach(() => {
  vi.resetModules();
  vi.unstubAllEnvs();
});

describe("Provider config zod schemas", () => {
  it("accepts a valid OpenAI config", () => {
    const parsed = OpenAIProviderConfigSchema.parse({
      provider: "openai",
      apiKey: "sk-test",
      organization: "org-x",
    });
    expect(parsed).toMatchObject({ provider: "openai", apiKey: "sk-test", organization: "org-x" });
  });

  it("accepts a valid Anthropic config", () => {
    const parsed = AnthropicProviderConfigSchema.parse({ provider: "anthropic", apiKey: "sk-ant" });
    expect(parsed.provider).toBe("anthropic");
  });

  it("accepts a valid Gemini config", () => {
    const parsed = GeminiProviderConfigSchema.parse({ provider: "gemini", apiKey: "ya29" });
    expect(parsed.provider).toBe("gemini");
  });

  it("requires resourceName for azure-openai", () => {
    expect(() =>
      AzureOpenAIProviderConfigSchema.parse({ provider: "azure-openai", apiKey: "az" }),
    ).toThrow();
    const parsed = AzureOpenAIProviderConfigSchema.parse({
      provider: "azure-openai",
      apiKey: "az",
      resourceName: "res",
    });
    expect(parsed.resourceName).toBe("res");
  });

  it("rejects empty apiKey", () => {
    expect(() => OpenAIProviderConfigSchema.parse({ provider: "openai", apiKey: "" })).toThrow();
  });

  it("rejects unknown provider via discriminated union", () => {
    expect(() => ProviderConfigSchema.parse({ provider: "cohere", apiKey: "k" })).toThrow();
  });

  it("keeps passthrough fields (Python `extra=allow` parity)", () => {
    const parsed = OpenAIProviderConfigSchema.parse({
      provider: "openai",
      apiKey: "sk-test",
      headers: { "x-custom": "1" },
    });
    expect((parsed as { headers?: unknown }).headers).toEqual({ "x-custom": "1" });
  });

  it("exposes all four supported providers", () => {
    expect(PROVIDERS.toSorted()).toEqual(["anthropic", "azure-openai", "gemini", "openai"]);
  });

  it("PROVIDER_SCHEMAS maps each provider to its schema", () => {
    expect(PROVIDER_SCHEMAS.openai).toBe(OpenAIProviderConfigSchema);
    expect(PROVIDER_SCHEMAS.anthropic).toBe(AnthropicProviderConfigSchema);
    expect(PROVIDER_SCHEMAS.gemini).toBe(GeminiProviderConfigSchema);
    expect(PROVIDER_SCHEMAS["azure-openai"]).toBe(AzureOpenAIProviderConfigSchema);
  });
});

describe("zod helpers", () => {
  it("parseProviderConfig throws on invalid input", () => {
    expect(() => parseProviderConfig({ provider: "openai" })).toThrow();
  });

  it("safeParseProviderConfig returns a SafeParseReturnType", () => {
    const ok = safeParseProviderConfig({ provider: "openai", apiKey: "k" });
    expect(ok.success).toBe(true);
    if (ok.success) expect(ok.data.provider).toBe("openai");

    const bad = safeParseProviderConfig({ provider: "openai", apiKey: "" });
    expect(bad.success).toBe(false);
    if (!bad.success) expect(bad.error.issues.length).toBeGreaterThan(0);
  });

  it("isProviderConfig narrows the type", () => {
    const candidate: unknown = { provider: "gemini", apiKey: "k" };
    if (isProviderConfig(candidate)) {
      expect(candidate.provider).toBe("gemini");
    } else {
      throw new Error("expected isProviderConfig to be true");
    }
    expect(isProviderConfig({ provider: "x" })).toBe(false);
  });
});

describe("createAdapter factory", () => {
  it("returns a VercelAIAdapter with the correct narrowed config", () => {
    const adapter = createAdapter({
      providerConfig: { provider: "openai", apiKey: "sk-test" },
      modelName: "gpt-4o-mini",
    });
    expect(adapter).toBeInstanceOf(VercelAIAdapter);
    expect(adapter.modelName).toBe("gpt-4o-mini");
    expect(adapter.providerConfig.provider).toBe("openai");
  });

  it("validates by default via zod", () => {
    expect(() =>
      createAdapter({
        providerConfig: { provider: "openai", apiKey: "" } as never,
        modelName: "gpt-4o-mini",
      }),
    ).toThrow();
  });

  it("skips validation when validate=false", () => {
    const adapter = createAdapter({
      providerConfig: { provider: "openai", apiKey: "anything" },
      modelName: "gpt-4o-mini",
      validate: false,
    });
    expect(adapter.providerConfig.apiKey).toBe("anything");
  });

  it("accepts each of the four supported providers", () => {
    const built = [
      createAdapter({ providerConfig: { provider: "openai", apiKey: "k" }, modelName: "m" }),
      createAdapter({ providerConfig: { provider: "anthropic", apiKey: "k" }, modelName: "m" }),
      createAdapter({ providerConfig: { provider: "gemini", apiKey: "k" }, modelName: "m" }),
      createAdapter({
        providerConfig: { provider: "azure-openai", apiKey: "k", resourceName: "r" },
        modelName: "m",
      }),
    ];
    expect(built.map((a) => a.providerConfig.provider)).toEqual([
      "openai",
      "anthropic",
      "gemini",
      "azure-openai",
    ]);
  });
});

describe("VercelAIAdapter.model() dispatches by provider", () => {
  it("calls createOpenAI with settings stripped of the discriminator", async () => {
    const { sentinel, modelFactory, providerFactory } = mockProviderFactory("openai-model");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter(
      { provider: "openai", apiKey: "sk-test", organization: "org-x" },
      "gpt-4o-mini",
    );
    const model = await adapter.model();

    expect(providerFactory).toHaveBeenCalledTimes(1);
    const settings = providerFactory.mock.calls[0]?.[0];
    expect(settings).toEqual({ apiKey: "sk-test", organization: "org-x" });
    expect(settings).not.toHaveProperty("provider");
    expect(modelFactory).toHaveBeenCalledWith("gpt-4o-mini");
    expect(model).toBe(sentinel);

    vi.doUnmock("@ai-sdk/openai");
  });

  it("calls createAnthropic for anthropic configs", async () => {
    const { sentinel, providerFactory } = mockProviderFactory("anthropic-model");
    vi.doMock("@ai-sdk/anthropic", () => ({ createAnthropic: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "anthropic", apiKey: "sk-ant" }, "claude-3");
    expect(await adapter.model()).toBe(sentinel);
    expect(providerFactory.mock.calls[0]?.[0]).toEqual({ apiKey: "sk-ant" });

    vi.doUnmock("@ai-sdk/anthropic");
  });

  it("calls createGoogleGenerativeAI for gemini configs", async () => {
    const { sentinel, providerFactory } = mockProviderFactory("gemini-model");
    vi.doMock("@ai-sdk/google", () => ({ createGoogleGenerativeAI: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "gemini", apiKey: "ya29" }, "gemini-2.0");
    expect(await adapter.model()).toBe(sentinel);

    vi.doUnmock("@ai-sdk/google");
  });

  it("calls createAzure for azure-openai configs", async () => {
    const { sentinel, providerFactory } = mockProviderFactory("azure-model");
    vi.doMock("@ai-sdk/azure", () => ({ createAzure: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter(
      { provider: "azure-openai", apiKey: "az", resourceName: "res" },
      "gpt-4o-mini",
    );
    expect(await adapter.model()).toBe(sentinel);
    expect(providerFactory.mock.calls[0]?.[0]).toEqual({ apiKey: "az", resourceName: "res" });

    vi.doUnmock("@ai-sdk/azure");
  });

  it("memoizes the model across calls", async () => {
    const { providerFactory } = mockProviderFactory("openai-model");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk-test" }, "gpt-4o-mini");
    const a = await adapter.model();
    const b = await adapter.model();
    expect(a).toBe(b);
    expect(providerFactory).toHaveBeenCalledTimes(1);

    vi.doUnmock("@ai-sdk/openai");
  });

  it("concurrent model() calls resolve from a single build", async () => {
    const { sentinel, providerFactory } = mockProviderFactory("openai-concurrent");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk" }, "gpt-4o-mini");
    const [a, b, c] = await Promise.all([adapter.model(), adapter.model(), adapter.model()]);
    expect(a).toBe(sentinel);
    expect(b).toBe(sentinel);
    expect(c).toBe(sentinel);
    expect(providerFactory).toHaveBeenCalledTimes(1);

    vi.doUnmock("@ai-sdk/openai");
  });

  it("resets the model cache when build fails so the next call can retry", async () => {
    const sentinel = Symbol("late-success");
    let callCount = 0;
    const providerFactory = vi.fn<ProviderFactory>(() => {
      callCount += 1;
      if (callCount === 1) throw new TypeError("transient failure");
      return () => sentinel;
    });
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk" }, "gpt-4o-mini");
    await expect(adapter.model()).rejects.toBeInstanceOf(TypeError);
    expect(await adapter.model()).toBe(sentinel);
    expect(providerFactory).toHaveBeenCalledTimes(2);

    vi.doUnmock("@ai-sdk/openai");
  });
});

describe("isModuleNotFound predicate", () => {
  it("matches the standard Node module-not-found codes", async () => {
    const { isModuleNotFound } = await import("../src/adapter");
    expect(isModuleNotFound(Object.assign(new Error("x"), { code: "ERR_MODULE_NOT_FOUND" }))).toBe(true);
    expect(isModuleNotFound(Object.assign(new Error("x"), { code: "MODULE_NOT_FOUND" }))).toBe(true);
  });

  it("falls back to message inspection when no code is present", async () => {
    const { isModuleNotFound } = await import("../src/adapter");
    expect(isModuleNotFound(new Error("Cannot find module '@ai-sdk/openai'"))).toBe(true);
    expect(isModuleNotFound(new Error("Cannot find package 'ai'"))).toBe(true);
  });

  it("does not match unrelated errors", async () => {
    const { isModuleNotFound } = await import("../src/adapter");
    expect(isModuleNotFound(new SyntaxError("oops"))).toBe(false);
    expect(isModuleNotFound(new TypeError("bad type"))).toBe(false);
    expect(isModuleNotFound("not an error")).toBe(false);
    expect(isModuleNotFound(null)).toBe(false);
  });
});

describe("VercelAIAdapter.generate / stream", () => {
  it("generate() forwards options + resolved model to ai.generateText", async () => {
    const { providerFactory } = mockProviderFactory("openai-generate");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const sdkResult = { text: "hello world", usage: { inputTokens: 3, outputTokens: 2 } };
    type AnyFn = (args: Record<string, unknown>) => unknown;
    const generateText = vi.fn<AnyFn>(async () => sdkResult);
    const streamText = vi.fn<AnyFn>();
    vi.doMock("ai", () => ({ generateText, streamText }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk" }, "gpt-4o-mini");
    const result = await adapter.generate({ prompt: "hi", system: "be brief" });

    expect(result).toBe(sdkResult);
    expect(generateText).toHaveBeenCalledTimes(1);
    const callArgs = generateText.mock.calls[0]?.[0];
    expect(callArgs).toMatchObject({ prompt: "hi", system: "be brief" });
    expect(callArgs?.model).toBeDefined();

    vi.doUnmock("ai");
    vi.doUnmock("@ai-sdk/openai");
  });

  it("stream() forwards options + resolved model to ai.streamText", async () => {
    const { providerFactory } = mockProviderFactory("openai-stream");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    const sdkResult = { textStream: (async function* () { yield "a"; yield "b"; })(), usage: Promise.resolve({ inputTokens: 1, outputTokens: 2 }) };
    type AnyFn = (args: Record<string, unknown>) => unknown;
    const generateText = vi.fn<AnyFn>();
    const streamText = vi.fn<AnyFn>(() => sdkResult);
    vi.doMock("ai", () => ({ generateText, streamText }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk" }, "gpt-4o-mini");
    const result = await adapter.stream({ prompt: "hi" });

    expect(result).toBe(sdkResult);
    expect(streamText).toHaveBeenCalledTimes(1);
    const callArgs = streamText.mock.calls[0]?.[0];
    expect(callArgs).toMatchObject({ prompt: "hi" });

    vi.doUnmock("ai");
    vi.doUnmock("@ai-sdk/openai");
  });

  it("ai SDK module is loaded only once across multiple convenience calls", async () => {
    const { providerFactory } = mockProviderFactory("openai-memo");
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI: providerFactory }));

    let aiLoads = 0;
    const generateText = vi.fn(async () => ({ text: "x" }));
    vi.doMock("ai", () => {
      aiLoads += 1;
      return { generateText, streamText: vi.fn() };
    });

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk" }, "gpt-4o-mini");
    await adapter.generate({ prompt: "1" });
    await adapter.generate({ prompt: "2" });
    await adapter.generate({ prompt: "3" });
    expect(aiLoads).toBe(1);
    expect(generateText).toHaveBeenCalledTimes(3);

    vi.doUnmock("ai");
    vi.doUnmock("@ai-sdk/openai");
  });
});
