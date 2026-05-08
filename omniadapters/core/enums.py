from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure-openai"


class Capability(StrEnum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"


class Model(StrEnum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5-20251101"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251015"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_FLASH = "gemini-3-flash"


MODEL_TO_PROVIDER: dict[Model, Provider] = {
    Model.GPT_4O: Provider.OPENAI,
    Model.GPT_4O_MINI: Provider.OPENAI,
    Model.O3_MINI: Provider.OPENAI,
    Model.O4_MINI: Provider.OPENAI,
    Model.GPT_5_4: Provider.OPENAI,
    Model.GPT_5_4_MINI: Provider.OPENAI,
    Model.GPT_5_4_NANO: Provider.OPENAI,
    Model.CLAUDE_SONNET_4_5: Provider.ANTHROPIC,
    Model.CLAUDE_OPUS_4_5: Provider.ANTHROPIC,
    Model.CLAUDE_HAIKU_4_5: Provider.ANTHROPIC,
    Model.CLAUDE_SONNET_4_6: Provider.ANTHROPIC,
    Model.CLAUDE_OPUS_4_7: Provider.ANTHROPIC,
    Model.GEMINI_2_5_PRO: Provider.GEMINI,
    Model.GEMINI_2_5_FLASH: Provider.GEMINI,
    Model.GEMINI_2_5_FLASH_LITE: Provider.GEMINI,
    Model.GEMINI_3_PRO: Provider.GEMINI,
    Model.GEMINI_3_FLASH: Provider.GEMINI,
}


def infer_provider(model: Model | str) -> Provider | None:
    if isinstance(model, Model):
        return MODEL_TO_PROVIDER.get(model)

    model_lower = model.lower()
    if "gpt" in model_lower or model_lower.startswith(("o1", "o3", "o4")):
        return Provider.OPENAI
    if "claude" in model_lower:
        return Provider.ANTHROPIC
    if "gemini" in model_lower:
        return Provider.GEMINI
    return None
