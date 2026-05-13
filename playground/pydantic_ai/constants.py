from __future__ import annotations

ENV_FILE_VAR = "PYDANTIC_AI_DEMO_ENV_FILE"
YAML_FILE_VAR = "PYDANTIC_AI_DEMO_YAML_FILE"

APP_TITLE = "pydantic-ai demo"

# NOTE: omniadapters' Provider enum names `gemini` and `azure-openai` do not
# match the names pydantic-ai's `infer_provider_class` expects
# (`google-gla`/`google-vertex` and `azure`), so those providers cannot be
# wired through `PydanticAIAdapter` today. Reject them at config-load time.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic"})
