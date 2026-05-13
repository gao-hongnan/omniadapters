import { z } from "zod";

export const PROVIDERS = ["openai", "anthropic", "gemini", "azure-openai"] as const;
export type Provider = (typeof PROVIDERS)[number];

const baseFields = {
  apiKey: z.string().min(1, "apiKey is required"),
};

export const OpenAIProviderConfigSchema = z
  .object({
    provider: z.literal("openai"),
    baseURL: z.string().url().optional(),
    organization: z.string().optional(),
    project: z.string().optional(),
    ...baseFields,
  })
  .passthrough();

export const AnthropicProviderConfigSchema = z
  .object({
    provider: z.literal("anthropic"),
    baseURL: z.string().url().optional(),
    ...baseFields,
  })
  .passthrough();

export const GeminiProviderConfigSchema = z
  .object({
    provider: z.literal("gemini"),
    baseURL: z.string().url().optional(),
    ...baseFields,
  })
  .passthrough();

export const AzureOpenAIProviderConfigSchema = z
  .object({
    provider: z.literal("azure-openai"),
    resourceName: z.string().min(1, "resourceName is required for azure-openai"),
    apiVersion: z.string().optional(),
    baseURL: z.string().url().optional(),
    ...baseFields,
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
