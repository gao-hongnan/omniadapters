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

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export type GenerateOptions = DistributiveOmit<GenerateTextOptions, "model">;
export type StreamOptions = DistributiveOmit<StreamTextOptions, "model">;

export type GenerateResult = Awaited<ReturnType<typeof generateText>>;
export type StreamResult = ReturnType<typeof streamText>;

type ProviderSettings<C extends ProviderConfig> = Omit<C, "provider">;

function stripDiscriminator<C extends ProviderConfig>(config: C): ProviderSettings<C> {
  const { provider: _provider, ...settings } = config;
  return settings as ProviderSettings<C>;
}

export function isModuleNotFound(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const code = (err as NodeJS.ErrnoException).code;
  if (code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND") return true;
  return /Cannot find (module|package)/i.test(err.message);
}

async function importOrFail<T>(specifier: string, errorMessage: string): Promise<T> {
  try {
    return (await import(specifier)) as T;
  } catch (cause) {
    if (!isModuleNotFound(cause)) throw cause;
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
    default: {
      const _exhaustive: never = config;
      throw new Error(`Unhandled provider config: ${JSON.stringify(_exhaustive)}`);
    }
  }
}

export class VercelAIAdapter<C extends ProviderConfig = ProviderConfig> {
  private _modelPromise: Promise<LanguageModel> | null = null;
  private _aiSdkPromise: Promise<typeof import("ai")> | null = null;

  constructor(
    public readonly providerConfig: C,
    public readonly modelName: string,
  ) {}

  model(): Promise<LanguageModel> {
    if (this._modelPromise === null) {
      this._modelPromise = buildModel(this.providerConfig, this.modelName).catch((err) => {
        this._modelPromise = null;
        throw err;
      });
    }
    return this._modelPromise;
  }

  private loadAiSdk(): Promise<typeof import("ai")> {
    if (this._aiSdkPromise === null) {
      this._aiSdkPromise = importOrFail<typeof import("ai")>("ai", AI_SDK_IMPORT_ERROR).catch(
        (err) => {
          this._aiSdkPromise = null;
          throw err;
        },
      );
    }
    return this._aiSdkPromise;
  }

  async generate(options: GenerateOptions): Promise<GenerateResult> {
    const [{ generateText }, model] = await Promise.all([this.loadAiSdk(), this.model()]);
    return generateText({ ...options, model } as GenerateTextOptions);
  }

  async stream(options: StreamOptions): Promise<StreamResult> {
    const [{ streamText }, model] = await Promise.all([this.loadAiSdk(), this.model()]);
    return streamText({ ...options, model } as StreamTextOptions);
  }
}
