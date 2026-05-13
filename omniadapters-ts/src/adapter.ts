import type {
  CoreMessage,
  GenerateTextResult,
  LanguageModel,
  StreamTextResult,
  generateText,
  streamText,
} from "ai";

import type { ProviderConfig } from "./core/config";
import {
  ANTHROPIC_IMPORT_ERROR,
  AZURE_OPENAI_IMPORT_ERROR,
  AI_SDK_IMPORT_ERROR,
  GEMINI_IMPORT_ERROR,
  MissingProviderPackageError,
  OPENAI_IMPORT_ERROR,
} from "./core/errors";

type GenerateTextParams = Parameters<typeof generateText>[0];
type StreamTextParams = Parameters<typeof streamText>[0];

export type GenerateOptions = Omit<GenerateTextParams, "model">;
export type StreamOptions = Omit<StreamTextParams, "model">;

async function loadAiSdk(): Promise<{
  generateText: typeof generateText;
  streamText: typeof streamText;
}> {
  try {
    const mod = await import("ai");
    return { generateText: mod.generateText, streamText: mod.streamText };
  } catch (err) {
    throw new MissingProviderPackageError(AI_SDK_IMPORT_ERROR, err);
  }
}

async function buildOpenAIModel(
  config: Extract<ProviderConfig, { provider: "openai" }>,
  modelName: string,
): Promise<LanguageModel> {
  let createOpenAI: (settings: Record<string, unknown>) => (id: string) => LanguageModel;
  try {
    ({ createOpenAI } = (await import("@ai-sdk/openai")) as {
      createOpenAI: typeof createOpenAI;
    });
  } catch (err) {
    throw new MissingProviderPackageError(OPENAI_IMPORT_ERROR, err);
  }
  const { provider: _provider, ...settings } = config;
  return createOpenAI(settings)(modelName);
}

async function buildAnthropicModel(
  config: Extract<ProviderConfig, { provider: "anthropic" }>,
  modelName: string,
): Promise<LanguageModel> {
  let createAnthropic: (settings: Record<string, unknown>) => (id: string) => LanguageModel;
  try {
    ({ createAnthropic } = (await import("@ai-sdk/anthropic")) as {
      createAnthropic: typeof createAnthropic;
    });
  } catch (err) {
    throw new MissingProviderPackageError(ANTHROPIC_IMPORT_ERROR, err);
  }
  const { provider: _provider, ...settings } = config;
  return createAnthropic(settings)(modelName);
}

async function buildGeminiModel(
  config: Extract<ProviderConfig, { provider: "gemini" }>,
  modelName: string,
): Promise<LanguageModel> {
  let createGoogleGenerativeAI: (
    settings: Record<string, unknown>,
  ) => (id: string) => LanguageModel;
  try {
    ({ createGoogleGenerativeAI } = (await import("@ai-sdk/google")) as {
      createGoogleGenerativeAI: typeof createGoogleGenerativeAI;
    });
  } catch (err) {
    throw new MissingProviderPackageError(GEMINI_IMPORT_ERROR, err);
  }
  const { provider: _provider, ...settings } = config;
  return createGoogleGenerativeAI(settings)(modelName);
}

async function buildAzureOpenAIModel(
  config: Extract<ProviderConfig, { provider: "azure-openai" }>,
  modelName: string,
): Promise<LanguageModel> {
  let createAzure: (settings: Record<string, unknown>) => (id: string) => LanguageModel;
  try {
    ({ createAzure } = (await import("@ai-sdk/azure")) as {
      createAzure: typeof createAzure;
    });
  } catch (err) {
    throw new MissingProviderPackageError(AZURE_OPENAI_IMPORT_ERROR, err);
  }
  const { provider: _provider, ...settings } = config;
  return createAzure(settings)(modelName);
}

export class VercelAIAdapter<C extends ProviderConfig = ProviderConfig> {
  private _model: LanguageModel | null = null;

  constructor(
    public readonly providerConfig: C,
    public readonly modelName: string,
  ) {}

  async model(): Promise<LanguageModel> {
    if (this._model !== null) return this._model;

    const config = this.providerConfig as ProviderConfig;
    switch (config.provider) {
      case "openai":
        this._model = await buildOpenAIModel(config, this.modelName);
        break;
      case "anthropic":
        this._model = await buildAnthropicModel(config, this.modelName);
        break;
      case "gemini":
        this._model = await buildGeminiModel(config, this.modelName);
        break;
      case "azure-openai":
        this._model = await buildAzureOpenAIModel(config, this.modelName);
        break;
      default: {
        const _exhaustive: never = config;
        throw new Error(`Unknown provider in config: ${JSON.stringify(_exhaustive)}`);
      }
    }
    return this._model;
  }

  async generate(
    messagesOrOptions: CoreMessage[] | GenerateOptions,
    options?: GenerateOptions,
  ): Promise<GenerateTextResult<Record<string, never>, never>> {
    const { generateText } = await loadAiSdk();
    const model = await this.model();
    const params = normalizeOptions(messagesOrOptions, options);
    return generateText({ model, ...params } as GenerateTextParams) as Promise<
      GenerateTextResult<Record<string, never>, never>
    >;
  }

  async stream(
    messagesOrOptions: CoreMessage[] | StreamOptions,
    options?: StreamOptions,
  ): Promise<StreamTextResult<Record<string, never>, never>> {
    const { streamText } = await loadAiSdk();
    const model = await this.model();
    const params = normalizeOptions(messagesOrOptions, options);
    return streamText({ model, ...params } as StreamTextParams) as StreamTextResult<
      Record<string, never>,
      never
    >;
  }
}

function normalizeOptions<O extends GenerateOptions | StreamOptions>(
  messagesOrOptions: CoreMessage[] | O,
  options: O | undefined,
): O {
  if (Array.isArray(messagesOrOptions)) {
    return { messages: messagesOrOptions, ...(options ?? {}) } as O;
  }
  return messagesOrOptions;
}
