"""Structured exception hierarchy for omniadapters.completion adapters.

Mirrors pydantic_ai's exception design (`pydantic_ai/exceptions.py:169-254`):
a small subclass hierarchy where each `except` clause narrows by class
identity. Vendor SDKs raise their own exception types (`openai.APIStatusError`,
`anthropic.APIStatusError`, `google.genai.errors.APIError`); the per-vendor
`_map_api_errors` context manager translates each into one of these unified
types so consumers catch a single vocabulary.

Three classes, in increasing specificity:

- `CompletionAdapterError`      base; carries `message`. Never raised directly.
- `CompletionAPIError`          API-call failure with no HTTP status (connection
                                refused, timeout, DNS failure). Carries `model_name`.
- `CompletionHTTPError`         HTTP 4xx/5xx response. Carries `status_code` and
                                vendor `body` for diagnostic logging.
"""

from __future__ import annotations

__all__ = ("CompletionAPIError", "CompletionAdapterError", "CompletionHTTPError")


class CompletionAdapterError(Exception):
    """Base class for all omniadapters.completion exceptions.

    Carries a single `message` attribute. Subclasses add structured fields.
    """

    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

    def __reduce__(self) -> tuple[type, tuple[object, ...]]:
        return self.__class__, (self.message,)


class CompletionAPIError(CompletionAdapterError):
    """API-call failure with no HTTP status code.

    Raised for transport-level failures such as connection refused, DNS
    resolution failure, TLS handshake errors, or client-side timeouts.
    """

    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        self.model_name = model_name
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[object, ...]]:
        return self.__class__, (self.model_name, self.message)


class CompletionHTTPError(CompletionAPIError):
    """HTTP 4xx/5xx response from the vendor API.

    Carries `status_code` (the HTTP status returned by the vendor) and
    `body` (the parsed response body, vendor-specific shape — typically a
    dict for OpenAI/Anthropic, a string or dict for Google).
    """

    status_code: int
    body: object | None

    def __init__(self, status_code: int, model_name: str, body: object | None = None) -> None:
        self.status_code = status_code
        self.body = body
        message = f"status_code: {status_code}, model_name: {model_name}, body: {body}"
        super().__init__(model_name, message)

    def __reduce__(self) -> tuple[type, tuple[object, ...]]:
        return self.__class__, (self.status_code, self.model_name, self.body)
