from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_ai.settings import ModelSettings
from pydantic_settings import SettingsConfigDict
from pydanticonf.settings import BaseSettingsWithYaml

from omniadapters.core.models import ProviderConfig
from omniadapters.pydantic_ai import create_adapter

from ..constants import SUPPORTED_PROVIDERS

if TYPE_CHECKING:
    from pydantic_ai import Agent


class AgentConfig(BaseModel):
    """Per-agent configuration for the pydantic-ai demo.

    -   https://ai.pydantic.dev/agents/
    -   https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings
    """

    provider_config: ProviderConfig
    model_name: str
    system_prompt: str = ""
    model_settings: ModelSettings | None = None

    # NOTE: arbitrary_types_allowed is needed because ModelSettings carries a
    # `timeout: float | httpx.Timeout` field that Pydantic cannot auto-schema.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_provider_supported(self) -> Self:
        provider = self.provider_config.provider
        if provider not in SUPPORTED_PROVIDERS:
            msg = (
                f"provider '{provider}' is not supported by this demo; pydantic-ai's "
                f"infer_provider_class expects different names for Google "
                f"('google-gla'/'google-vertex') and Azure-OpenAI ('azure'). "
                f"Supported here: {sorted(SUPPORTED_PROVIDERS)}."
            )
            raise ValueError(msg)
        return self

    def build_agent(self) -> Agent[None, str]:
        adapter = create_adapter(provider_config=self.provider_config, model_name=self.model_name)
        return adapter.create_agent(
            system_prompt=self.system_prompt,
            model_settings=self.model_settings,
        )


class PydanticAIDemoConfig(BaseModel):
    agents: dict[str, AgentConfig]


class Settings(BaseSettingsWithYaml):  # NOTE: must subclass; see playground/critic/config/settings.py:53
    pydantic_ai_demo: PydanticAIDemoConfig

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
