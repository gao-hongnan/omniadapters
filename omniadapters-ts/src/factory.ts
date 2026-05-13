import { VercelAIAdapter } from "./adapter";
import {
  type AnthropicProviderConfig,
  type AzureOpenAIProviderConfig,
  type GeminiProviderConfig,
  type OpenAIProviderConfig,
  type ProviderConfig,
  ProviderConfigSchema,
} from "./core/config";

export interface CreateAdapterArgs<C extends ProviderConfig = ProviderConfig> {
  providerConfig: C;
  modelName: string;
  validate?: boolean;
}

export function createAdapter(
  args: CreateAdapterArgs<OpenAIProviderConfig>,
): VercelAIAdapter<OpenAIProviderConfig>;
export function createAdapter(
  args: CreateAdapterArgs<AnthropicProviderConfig>,
): VercelAIAdapter<AnthropicProviderConfig>;
export function createAdapter(
  args: CreateAdapterArgs<GeminiProviderConfig>,
): VercelAIAdapter<GeminiProviderConfig>;
export function createAdapter(
  args: CreateAdapterArgs<AzureOpenAIProviderConfig>,
): VercelAIAdapter<AzureOpenAIProviderConfig>;
export function createAdapter(
  args: CreateAdapterArgs<ProviderConfig>,
): VercelAIAdapter<ProviderConfig>;
export function createAdapter(args: CreateAdapterArgs): VercelAIAdapter {
  const config = args.validate === false
    ? args.providerConfig
    : (ProviderConfigSchema.parse(args.providerConfig) as ProviderConfig);
  return new VercelAIAdapter(config, args.modelName);
}
