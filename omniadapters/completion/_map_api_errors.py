"""Per-vendor `_map_api_errors` context managers.

Each function is a `@contextmanager` that wraps a single vendor API call and
translates vendor SDK exceptions into the unified `omniadapters.completion`
exception hierarchy. Vendor SDKs are imported lazily inside each function
body so this module can be imported without any vendor SDK installed —
import failures surface only at call time, matching the existing per-adapter
pattern in `adapters/openai.py`.

Mirrors `pydantic_ai/models/openai.py:164-173`,
`pydantic_ai/models/anthropic.py:242-251`, and
`pydantic_ai/models/google.py:738-747` directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from .errors import CompletionAPIError, CompletionHTTPError

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = (
    "_map_anthropic_errors",
    "_map_azure_openai_errors",
    "_map_google_errors",
    "_map_openai_errors",
)

_HTTP_ERROR_THRESHOLD = 400


@contextmanager
def _map_openai_errors(*, model_name: str) -> Generator[None]:
    """Translate openai SDK exceptions into the completion exception hierarchy."""
    from openai import APIConnectionError, APIStatusError

    try:
        yield
    except APIStatusError as e:
        if (status_code := e.status_code) >= _HTTP_ERROR_THRESHOLD:
            raise CompletionHTTPError(status_code, model_name, e.body) from e
        raise CompletionAPIError(model_name, e.message) from e
    except APIConnectionError as e:
        raise CompletionAPIError(model_name, e.message) from e


@contextmanager
def _map_anthropic_errors(*, model_name: str) -> Generator[None]:
    """Translate anthropic SDK exceptions into the completion exception hierarchy."""
    from anthropic import APIConnectionError, APIStatusError

    try:
        yield
    except APIStatusError as e:
        if (status_code := e.status_code) >= _HTTP_ERROR_THRESHOLD:
            raise CompletionHTTPError(status_code, model_name, e.body) from e
        raise CompletionAPIError(model_name, e.message) from e
    except APIConnectionError as e:
        raise CompletionAPIError(model_name, e.message) from e


@contextmanager
def _map_google_errors(*, model_name: str) -> Generator[None]:
    """Translate google.genai SDK exceptions into the completion exception hierarchy.

    `google.genai.errors.APIError` uses `.code` (int) instead of `.status_code`
    and `.details` instead of `.body`. There is no separate connection-error
    class — connection failures surface as `APIError` with `code=0` or via
    httpx exceptions that bubble through unchanged.
    """
    from google.genai.errors import APIError

    try:
        yield
    except APIError as e:
        if (status_code := e.code) >= _HTTP_ERROR_THRESHOLD:
            raise CompletionHTTPError(status_code, model_name, cast("object", e.details)) from e
        raise CompletionAPIError(model_name, str(e)) from e


@contextmanager
def _map_azure_openai_errors(*, model_name: str) -> Generator[None]:
    """Translate Azure OpenAI exceptions — same `openai.*` classes as OpenAI."""
    with _map_openai_errors(model_name=model_name):
        yield
