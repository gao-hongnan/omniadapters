import { describe, expect, it, vi } from "vitest";

import {
  AnthropicProviderConfigSchema,
  AzureOpenAIProviderConfigSchema,
  GeminiProviderConfigSchema,
  OpenAIProviderConfigSchema,
  PROVIDERS,
  ProviderConfigSchema,
  VercelAIAdapter,
  createAdapter,
} from "../src/index";

describe("ProviderConfig schemas", () => {
  it("accepts a valid OpenAI config", () => {
    const parsed = OpenAIProviderConfigSchema.parse({
      provider: "openai",
      apiKey: "sk-test",
    });
    expect(parsed.provider).toBe("openai");
    expect(parsed.apiKey).toBe("sk-test");
  });

  it("accepts a valid Anthropic config", () => {
    const parsed = AnthropicProviderConfigSchema.parse({
      provider: "anthropic",
      apiKey: "sk-ant-test",
    });
    expect(parsed.provider).toBe("anthropic");
  });

  it("accepts a valid Gemini config", () => {
    const parsed = GeminiProviderConfigSchema.parse({
      provider: "gemini",
      apiKey: "ya29.test",
    });
    expect(parsed.provider).toBe("gemini");
  });

  it("requires resourceName for azure-openai", () => {
    expect(() =>
      AzureOpenAIProviderConfigSchema.parse({
        provider: "azure-openai",
        apiKey: "az-test",
      }),
    ).toThrow();

    const parsed = AzureOpenAIProviderConfigSchema.parse({
      provider: "azure-openai",
      apiKey: "az-test",
      resourceName: "my-resource",
    });
    expect(parsed.resourceName).toBe("my-resource");
  });

  it("rejects empty apiKey", () => {
    expect(() => OpenAIProviderConfigSchema.parse({ provider: "openai", apiKey: "" })).toThrow();
  });

  it("rejects unknown provider via discriminated union", () => {
    expect(() =>
      ProviderConfigSchema.parse({ provider: "cohere", apiKey: "k" } as unknown),
    ).toThrow();
  });

  it("exposes all four supported providers", () => {
    expect([...PROVIDERS].sort()).toEqual(["anthropic", "azure-openai", "gemini", "openai"]);
  });
});

describe("createAdapter factory", () => {
  it("returns a VercelAIAdapter instance with correct config", () => {
    const adapter = createAdapter({
      providerConfig: { provider: "openai", apiKey: "sk-test" },
      modelName: "gpt-4o-mini",
    });
    expect(adapter).toBeInstanceOf(VercelAIAdapter);
    expect(adapter.modelName).toBe("gpt-4o-mini");
    expect(adapter.providerConfig.provider).toBe("openai");
  });

  it("validates by default", () => {
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

  it("accepts each of the four providers", () => {
    const openai = createAdapter({
      providerConfig: { provider: "openai", apiKey: "k" },
      modelName: "m",
    });
    const anthropic = createAdapter({
      providerConfig: { provider: "anthropic", apiKey: "k" },
      modelName: "m",
    });
    const gemini = createAdapter({
      providerConfig: { provider: "gemini", apiKey: "k" },
      modelName: "m",
    });
    const azure = createAdapter({
      providerConfig: { provider: "azure-openai", apiKey: "k", resourceName: "r" },
      modelName: "m",
    });
    expect(openai.providerConfig.provider).toBe("openai");
    expect(anthropic.providerConfig.provider).toBe("anthropic");
    expect(gemini.providerConfig.provider).toBe("gemini");
    expect(azure.providerConfig.provider).toBe("azure-openai");
  });
});

describe("VercelAIAdapter.model() dispatches by provider", () => {
  it("calls createOpenAI with the provider settings (excluding the discriminator)", async () => {
    const sentinel = Symbol("openai-model");
    const factory = vi.fn<(id: string) => unknown>(() => sentinel);
    const createOpenAI = vi.fn<(settings: Record<string, unknown>) => typeof factory>(() => factory);
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter(
      { provider: "openai", apiKey: "sk-test", organization: "org-x" },
      "gpt-4o-mini",
    );
    const model = await adapter.model();

    expect(createOpenAI).toHaveBeenCalledTimes(1);
    const settings = createOpenAI.mock.calls[0]?.[0];
    expect(settings).toMatchObject({ apiKey: "sk-test", organization: "org-x" });
    expect(settings?.provider).toBeUndefined();
    expect(factory).toHaveBeenCalledWith("gpt-4o-mini");
    expect(model).toBe(sentinel);

    vi.doUnmock("@ai-sdk/openai");
    vi.resetModules();
  });

  it("calls createAzure for azure-openai", async () => {
    const sentinel = Symbol("azure-model");
    const factory = vi.fn<(id: string) => unknown>(() => sentinel);
    const createAzure = vi.fn<(settings: Record<string, unknown>) => typeof factory>(() => factory);
    vi.doMock("@ai-sdk/azure", () => ({ createAzure }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter(
      { provider: "azure-openai", apiKey: "az", resourceName: "res" },
      "gpt-4o-mini",
    );
    const model = await adapter.model();

    expect(createAzure).toHaveBeenCalledTimes(1);
    const settings = createAzure.mock.calls[0]?.[0];
    expect(settings).toMatchObject({ apiKey: "az", resourceName: "res" });
    expect(settings?.provider).toBeUndefined();
    expect(model).toBe(sentinel);

    vi.doUnmock("@ai-sdk/azure");
    vi.resetModules();
  });

  it("memoizes the model across calls", async () => {
    const sentinel = Symbol("openai-model");
    const factory = vi.fn<(id: string) => unknown>(() => sentinel);
    const createOpenAI = vi.fn<(settings: Record<string, unknown>) => typeof factory>(() => factory);
    vi.doMock("@ai-sdk/openai", () => ({ createOpenAI }));

    const { VercelAIAdapter: Adapter } = await import("../src/adapter");
    const adapter = new Adapter({ provider: "openai", apiKey: "sk-test" }, "gpt-4o-mini");
    const a = await adapter.model();
    const b = await adapter.model();

    expect(a).toBe(b);
    expect(createOpenAI).toHaveBeenCalledTimes(1);

    vi.doUnmock("@ai-sdk/openai");
    vi.resetModules();
  });
});
