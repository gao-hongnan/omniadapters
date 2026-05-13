import { z } from "zod";

export const PROVIDERS = ["openai", "anthropic", "gemini", "azure-openai"] as const;

export const ProviderSchema = z.enum(PROVIDERS);
export type Provider = z.infer<typeof ProviderSchema>;

const apiKeyField = z.string().min(1, "apiKey is required");
const optionalUrl = z.string().url().optional();

export const OpenAIProviderConfigSchema = z
  .object({
    provider: z.literal("openai"),
    apiKey: apiKeyField,
    baseURL: optionalUrl,
    organization: z.string().min(1).optional(),
    project: z.string().min(1).optional(),
  })
  .passthrough();

export const AnthropicProviderConfigSchema = z
  .object({
    provider: z.literal("anthropic"),
    apiKey: apiKeyField,
    baseURL: optionalUrl,
  })
  .passthrough();

export const GeminiProviderConfigSchema = z
  .object({
    provider: z.literal("gemini"),
    apiKey: apiKeyField,
    baseURL: optionalUrl,
  })
  .passthrough();

export const AzureOpenAIProviderConfigSchema = z
  .object({
    provider: z.literal("azure-openai"),
    apiKey: apiKeyField,
    resourceName: z.string().min(1, "resourceName is required for azure-openai"),
    apiVersion: z.string().min(1).optional(),
    baseURL: optionalUrl,
  })
  .passthrough();

export const ProviderConfigSchema = z.discriminatedUnion("provider", [
  OpenAIProviderConfigSchema,
  AnthropicProviderConfigSchema,
  GeminiProviderConfigSchema,
  AzureOpenAIProviderConfigSchema,
]);

export type OpenAIProviderConfig = z.infer<typeof OpenAIProviderConfigSchema>;
export type AnthropicProviderConfig = z.infer<typeof AnthropicProviderConfigSchema>;
export type GeminiProviderConfig = z.infer<typeof GeminiProviderConfigSchema>;
export type AzureOpenAIProviderConfig = z.infer<typeof AzureOpenAIProviderConfigSchema>;
export type ProviderConfig = z.infer<typeof ProviderConfigSchema>;

export type ProviderConfigByName<P extends Provider> = Extract<ProviderConfig, { provider: P }>;

export const PROVIDER_SCHEMAS: { [P in Provider]: z.ZodType<ProviderConfigByName<P>> } = {
  openai: OpenAIProviderConfigSchema,
  anthropic: AnthropicProviderConfigSchema,
  gemini: GeminiProviderConfigSchema,
  "azure-openai": AzureOpenAIProviderConfigSchema,
};

export function parseProviderConfig(input: unknown): ProviderConfig {
  return ProviderConfigSchema.parse(input);
}

export function safeParseProviderConfig(
  input: unknown,
): z.SafeParseReturnType<unknown, ProviderConfig> {
  return ProviderConfigSchema.safeParse(input);
}

export function isProviderConfig(input: unknown): input is ProviderConfig {
  return ProviderConfigSchema.safeParse(input).success;
}
