"""Settings for the critic playground: the factual-QA recipe config from YAML, API keys from a .env file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from omniadapters.services.cove.recipes.factual_qa import (
    FactualQAConfig,  # noqa: TC001 - pydantic resolves it at runtime
)

if TYPE_CHECKING:
    from pathlib import Path


class Settings(BaseSettings):
    """``cove`` comes from the YAML file; each role's key from ``COVE__<ROLE>__PROVIDER_CONFIG__API_KEY``.

    Unknown top-level keys are ignored because a shared .env file may hold other tools' variables.
    Inside ``cove`` the recipe config forbids unknown keys, so a misspelt or outdated role name
    (critic's ``drafter``, ``skeptic``, ``fact_checker``) fails loudly instead of being dropped.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file_encoding="utf-8", extra="ignore")

    cove: FactualQAConfig


def load_settings(*, yaml_file: Path, env_file: Path | None) -> Settings:
    """Load ``yaml_file`` and overlay the environment and ``env_file`` on it.

    Top-level keys starting with ``x-`` hold YAML anchors (the docker-compose convention). They
    are templates for the roles below them, not settings, so they are dropped before validation.
    """
    document = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        msg = f"{yaml_file} must contain a mapping, got {type(document).__name__}"
        raise TypeError(msg)
    values = {key: value for key, value in document.items() if not str(key).startswith("x-")}
    # pydantic-settings takes `_env_file` at runtime; pyright's dataclass_transform __init__ lists only fields.
    return Settings(**values, _env_file=env_file)  # pyright: ignore[reportCallIssue]
