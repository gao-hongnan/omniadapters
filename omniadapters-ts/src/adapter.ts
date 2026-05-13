import type { generateText, LanguageModel, streamText } from "ai";

import type {
  AnthropicProviderConfig,
  AzureOpenAIProviderConfig,
  GeminiProviderConfig,
  OpenAIProviderConfig,
  ProviderConfig,
} from "./core/config";
import {
  AI_SDK_IMPORT_ERROR,
  ANTHROPIC_IMPORT_ERROR,
  AZURE_OPENAI_IMPORT_ERROR,
  GEMINI_IMPORT_ERROR,
  MissingProviderPackageError,
  OPENAI_IMPORT_ERROR,
} from "./core/errors";

type GenerateTextOptions = Parameters<typeof generateText>[0];
type StreamTextOptions = Parameters<typeof streamText>[0];

export type GenerateOptions = Omit<GenerateTextOptions, "model">;
export type StreamOptions = Omit<StreamTextOptions, "model">;

export type GenerateResult = Awaited<ReturnType<typeof generateText>>;
export type StreamResult = ReturnType<typeof streamText>;

type ProviderSettings<C extends ProviderConfig> = Omit<C, "provider">;

function stripDiscriminator<C extends ProviderConfig>(config: C): ProviderSettings<C> {
  const { provider: _provider, ...settings } = config;
  return settings;
}

async function importOrFail<T>(specifier: string, errorMessage: string): Promise<T> {
  try {
    return (await import(specifier)) as T;
  } catch (cause) {
    throw new MissingProviderPackageError(errorMessage, { cause });
  }
}

async function buildOpenAI(config: OpenAIProviderConfig, modelName: string): Promise<LanguageModel> {
  const { createOpenAI } = await importOrFail<typeof import("@ai-sdk/openai")>(
    "@ai-sdk/openai",
    OPENAI_IMPORT_ERROR,
  );
  return createOpenAI(stripDiscriminator(config))(modelName);
}

async function buildAnthropic(
  config: AnthropicProviderConfig,
  modelName: string,
): Promise<LanguageModel> {
  const { createAnthropic } = await importOrFail<typeof import("@ai-sdk/anthropic")>(
    "@ai-sdk/anthropic",
    ANTHROPIC_IMPORT_ERROR,
  );
  return createAnthropic(stripDiscriminator(config))(modelName);
}

async function buildGemini(
  config: GeminiProviderConfig,
  modelName: string,
): Promise<LanguageModel> {
  const { createGoogleGenerativeAI } = await importOrFail<typeof import("@ai-sdk/google")>(
    "@ai-sdk/google",
    GEMINI_IMPORT_ERROR,
  );
  return createGoogleGenerativeAI(stripDiscriminator(config))(modelName);
}

async function buildAzureOpenAI(
  config: AzureOpenAIProviderConfig,
  modelName: string,
): Promise<LanguageModel> {
  const { createAzure } = await importOrFail<typeof import("@ai-sdk/azure")>(
    "@ai-sdk/azure",
    AZURE_OPENAI_IMPORT_ERROR,
  );
  return createAzure(stripDiscriminator(config))(modelName);
}

async function buildModel(config: ProviderConfig, modelName: string): Promise<LanguageModel> {
  switch (config.provider) {
    case "openai":
      return buildOpenAI(config, modelName);
    case "anthropic":
      return buildAnthropic(config, modelName);
    case "gemini":
      return buildGemini(config, modelName);
    case "azure-openai":
      return buildAzureOpenAI(config, modelName);
  }
}

async function loadAiSdk(): Promise<typeof import("ai")> {
  return importOrFail<typeof import("ai")>("ai", AI_SDK_IMPORT_ERROR);
}

export class VercelAIAdapter<C extends ProviderConfig = ProviderConfig> {
  private _model: LanguageModel | null = null;

  constructor(
    public readonly providerConfig: C,
    public readonly modelName: string,
  ) {}

  async model(): Promise<LanguageModel> {
    if (this._model === null) {
      this._model = await buildModel(this.providerConfig, this.modelName);
    }
    return this._model;
  }

  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const { generateText } = await loadAiSdk();
    const model = await this.model();
    return generateText({ ...options, model });
  }

  async stream(options: StreamOptions): Promise<StreamResult> {
    const { streamText } = await loadAiSdk();
    const model = await this.model();
    return streamText({ ...options, model });
  }
}
