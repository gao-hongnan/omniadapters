from __future__ import annotations

from functools import lru_cache
from typing import Any, Self

from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from pydanticonf.settings import BaseSettingsWithYaml

from omniadapters.core.models import CompletionClientParams, ProviderConfig
from omniadapters.structify.models import InstructorConfig


class PromptsConfig(BaseModel):
    base_path: str

    user_prompt_path: str
    user_context_variables: dict[str, Any] = {}

    system_prompt_path: str
    system_context_variables: dict[str, Any] = {}


class ProviderAgnosticAgent(BaseModel):
    """Fully typed agent config supporting multiple providers."""

    prompts: PromptsConfig
    provider_config: ProviderConfig
    completion_params: CompletionClientParams
    instructor_config: InstructorConfig

    @model_validator(mode="after")
    def validate_provider_match(self) -> Self:
        """Ensure provider types match."""
        if self.provider_config.provider != self.completion_params.provider:
            raise ValueError(f"Provider mismatch: {self.provider_config.provider} != {self.completion_params.provider}")
        return self


class CoVeVerifierConfig(BaseModel):
    """Configuration for Chain-of-Verification (CoVe) verifier.

    -   https://python.useinstructor.com/prompting/self_criticism/chain_of_verification/
    -   https://arxiv.org/pdf/2309.11495
    """

    drafter: ProviderAgnosticAgent
    skeptic: ProviderAgnosticAgent
    fact_checker: ProviderAgnosticAgent
    judge: ProviderAgnosticAgent


class Settings(BaseSettingsWithYaml):  # NOTE: if you do not subclass this, you will face error.
    cove: CoVeVerifierConfig

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="allow")


@lru_cache(maxsize=128)
def get_settings(env_file: str | None = None, yaml_file: str | None = None, **kwargs: Any) -> Settings:
    config_dict = Settings.model_config.copy()

    if env_file:
        config_dict["env_file"] = env_file
    if yaml_file:
        config_dict["yaml_file"] = yaml_file

    config_dict.update(kwargs)  # type: ignore[typeddict-item]

    class RuntimeSettings(Settings):
        model_config = SettingsConfigDict(**config_dict)

    return RuntimeSettings(**kwargs)
