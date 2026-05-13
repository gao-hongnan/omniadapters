export const OPENAI_IMPORT_ERROR =
  "OpenAI provider requires '@ai-sdk/openai'. Install with: `npm install @ai-sdk/openai`";

export const ANTHROPIC_IMPORT_ERROR =
  "Anthropic provider requires '@ai-sdk/anthropic'. Install with: `npm install @ai-sdk/anthropic`";

export const GEMINI_IMPORT_ERROR =
  "Gemini provider requires '@ai-sdk/google'. Install with: `npm install @ai-sdk/google`";

export const AZURE_OPENAI_IMPORT_ERROR =
  "Azure OpenAI provider requires '@ai-sdk/azure'. Install with: `npm install @ai-sdk/azure`";

export const AI_SDK_IMPORT_ERROR =
  "Vercel AI SDK is required. Install with: `npm install ai`";

export class MissingProviderPackageError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "MissingProviderPackageError";
  }
}
