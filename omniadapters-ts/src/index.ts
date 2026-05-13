export { VercelAIAdapter } from "./adapter";
export type {
  GenerateOptions,
  GenerateResult,
  StreamOptions,
  StreamResult,
} from "./adapter";

export { createAdapter } from "./factory";
export type { CreateAdapterArgs } from "./factory";

export {
  AnthropicProviderConfigSchema,
  AzureOpenAIProviderConfigSchema,
  GeminiProviderConfigSchema,
  OpenAIProviderConfigSchema,
  PROVIDER_SCHEMAS,
  PROVIDERS,
  ProviderConfigSchema,
  ProviderSchema,
  isProviderConfig,
  parseProviderConfig,
  safeParseProviderConfig,
} from "./core/config";
export type {
  AnthropicProviderConfig,
  AzureOpenAIProviderConfig,
  GeminiProviderConfig,
  OpenAIProviderConfig,
  Provider,
  ProviderConfig,
  ProviderConfigByName,
} from "./core/config";

export {
  AI_SDK_IMPORT_ERROR,
  ANTHROPIC_IMPORT_ERROR,
  AZURE_OPENAI_IMPORT_ERROR,
  GEMINI_IMPORT_ERROR,
  MissingProviderPackageError,
  OPENAI_IMPORT_ERROR,
} from "./core/errors";
