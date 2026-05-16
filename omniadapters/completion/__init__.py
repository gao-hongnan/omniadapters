"""omniadapters.completion — multi-vendor completion adapter package.

Public exports:

- ``create_adapter``           : factory for OpenAI / Anthropic / Gemini / Azure adapters.
- ``CompletionAdapterError``   : base exception for adapter failures.
- ``CompletionAPIError``       : transport-level failure (no HTTP status).
- ``CompletionHTTPError``      : 4xx/5xx HTTP response from vendor.
"""

from __future__ import annotations

from .errors import (
    CompletionAdapterError,
    CompletionAPIError,
    CompletionHTTPError,
)
from .factory import create_adapter

__all__ = ("CompletionAPIError", "CompletionAdapterError", "CompletionHTTPError", "create_adapter")
