from __future__ import annotations

from unittest.mock import patch

import pytest

from omniadapters.completion._map_api_errors import (
    _map_anthropic_errors,
    _map_azure_openai_errors,
    _map_google_errors,
    _map_openai_errors,
)
from omniadapters.completion.errors import CompletionAPIError, CompletionHTTPError

_MODEL = "test-model"
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_INTERNAL_SERVER_ERROR = 500


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str = "err", body: object = None) -> None:
        self.status_code = status_code
        self.message = message
        self.body = body


class _FakeConnectionError(Exception):
    def __init__(self, message: str = "conn err") -> None:
        self.message = message


class _FakeGoogleAPIError(Exception):
    def __init__(self, code: int, details: object = "detail") -> None:
        self.code = code
        self.details = details

    def __str__(self) -> str:
        return f"google error {self.code}"


@pytest.mark.unit
class TestMapOpenAIErrors:
    def test_http_status_error_raises_completion_http_error(self) -> None:
        body = {"error": "rate_limit"}
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionHTTPError) as exc_info,
            _map_openai_errors(model_name=_MODEL),
        ):
            raise _FakeStatusError(_HTTP_TOO_MANY_REQUESTS, "rate limited", body)

        err = exc_info.value
        assert err.status_code == _HTTP_TOO_MANY_REQUESTS, "HTTP status must be preserved on CompletionHTTPError."
        assert err.model_name == _MODEL
        assert err.body is body

    def test_sub_400_status_error_raises_completion_api_error(self) -> None:
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionAPIError) as exc_info,
            _map_openai_errors(model_name=_MODEL),
        ):
            raise _FakeStatusError(399, "below threshold")

        assert not isinstance(exc_info.value, CompletionHTTPError), (
            "Status codes below 400 must not produce CompletionHTTPError."
        )
        assert exc_info.value.model_name == _MODEL

    def test_connection_error_raises_completion_api_error(self) -> None:
        connection_refused_msg = "connection refused"
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionAPIError) as exc_info,
            _map_openai_errors(model_name=_MODEL),
        ):
            raise _FakeConnectionError(connection_refused_msg)

        assert not isinstance(exc_info.value, CompletionHTTPError)
        assert exc_info.value.model_name == _MODEL

    def test_non_sdk_exception_passes_through(self) -> None:
        unexpected_msg = "unexpected"
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            pytest.raises(RuntimeError, match="unexpected"),
            _map_openai_errors(model_name=_MODEL),
        ):
            raise RuntimeError(unexpected_msg)

    def test_clean_yield_does_not_raise(self) -> None:
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            _map_openai_errors(model_name=_MODEL),
        ):
            pass


@pytest.mark.unit
class TestMapAnthropicErrors:
    def test_http_status_error_raises_completion_http_error(self) -> None:
        body = {"type": "error", "error": {"type": "rate_limit_error"}}
        with (
            patch("anthropic.APIStatusError", _FakeStatusError),
            patch("anthropic.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionHTTPError) as exc_info,
            _map_anthropic_errors(model_name=_MODEL),
        ):
            raise _FakeStatusError(_HTTP_TOO_MANY_REQUESTS, "rate limited", body)

        err = exc_info.value
        assert err.status_code == _HTTP_TOO_MANY_REQUESTS
        assert err.model_name == _MODEL
        assert err.body is body

    def test_sub_400_status_error_raises_completion_api_error(self) -> None:
        with (
            patch("anthropic.APIStatusError", _FakeStatusError),
            patch("anthropic.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionAPIError) as exc_info,
            _map_anthropic_errors(model_name=_MODEL),
        ):
            raise _FakeStatusError(399, "below threshold")

        assert not isinstance(exc_info.value, CompletionHTTPError)
        assert exc_info.value.model_name == _MODEL

    def test_connection_error_raises_completion_api_error(self) -> None:
        dns_failure_msg = "dns failure"
        with (
            patch("anthropic.APIStatusError", _FakeStatusError),
            patch("anthropic.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionAPIError) as exc_info,
            _map_anthropic_errors(model_name=_MODEL),
        ):
            raise _FakeConnectionError(dns_failure_msg)

        assert not isinstance(exc_info.value, CompletionHTTPError)
        assert exc_info.value.model_name == _MODEL

    def test_non_sdk_exception_passes_through(self) -> None:
        bad_input_msg = "bad input"
        with (
            patch("anthropic.APIStatusError", _FakeStatusError),
            patch("anthropic.APIConnectionError", _FakeConnectionError),
            pytest.raises(ValueError, match="bad input"),
            _map_anthropic_errors(model_name=_MODEL),
        ):
            raise ValueError(bad_input_msg)

    def test_clean_yield_does_not_raise(self) -> None:
        with (
            patch("anthropic.APIStatusError", _FakeStatusError),
            patch("anthropic.APIConnectionError", _FakeConnectionError),
            _map_anthropic_errors(model_name=_MODEL),
        ):
            pass


@pytest.mark.unit
class TestMapGoogleErrors:
    def test_http_api_error_raises_completion_http_error(self) -> None:
        details = {"message": "quota exceeded"}
        with (
            patch("google.genai.errors.APIError", _FakeGoogleAPIError),
            pytest.raises(CompletionHTTPError) as exc_info,
            _map_google_errors(model_name=_MODEL),
        ):
            raise _FakeGoogleAPIError(_HTTP_TOO_MANY_REQUESTS, details)

        err = exc_info.value
        assert err.status_code == _HTTP_TOO_MANY_REQUESTS
        assert err.model_name == _MODEL
        assert err.body is details

    def test_sub_400_api_error_raises_completion_api_error(self) -> None:
        connection_failure_msg = "connection failure"
        with (
            patch("google.genai.errors.APIError", _FakeGoogleAPIError),
            pytest.raises(CompletionAPIError) as exc_info,
            _map_google_errors(model_name=_MODEL),
        ):
            raise _FakeGoogleAPIError(0, connection_failure_msg)

        assert not isinstance(exc_info.value, CompletionHTTPError), (
            "Google APIError with code=0 (connection-level) must not produce CompletionHTTPError."
        )
        assert exc_info.value.model_name == _MODEL

    def test_non_sdk_exception_passes_through(self) -> None:
        missing_key_msg = "missing key"
        with (
            patch("google.genai.errors.APIError", _FakeGoogleAPIError),
            pytest.raises(KeyError),
            _map_google_errors(model_name=_MODEL),
        ):
            raise KeyError(missing_key_msg)

    def test_clean_yield_does_not_raise(self) -> None:
        with patch("google.genai.errors.APIError", _FakeGoogleAPIError), _map_google_errors(model_name=_MODEL):
            pass


@pytest.mark.unit
class TestMapAzureOpenAIErrors:
    def test_delegates_to_openai_error_mapping(self) -> None:
        body = {"error": {"code": "InternalServerError"}}
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            pytest.raises(CompletionHTTPError) as exc_info,
            _map_azure_openai_errors(model_name=_MODEL),
        ):
            raise _FakeStatusError(_HTTP_INTERNAL_SERVER_ERROR, "server error", body)

        err = exc_info.value
        assert err.status_code == _HTTP_INTERNAL_SERVER_ERROR, (
            "_map_azure_openai_errors must delegate to _map_openai_errors, "
            "producing CompletionHTTPError for 5xx responses."
        )
        assert err.model_name == _MODEL

    def test_clean_yield_does_not_raise(self) -> None:
        with (
            patch("openai.APIStatusError", _FakeStatusError),
            patch("openai.APIConnectionError", _FakeConnectionError),
            _map_azure_openai_errors(model_name=_MODEL),
        ):
            pass
