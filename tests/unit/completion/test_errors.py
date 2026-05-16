from __future__ import annotations

import pytest

from omniadapters.completion.errors import (
    CompletionAdapterError,
    CompletionAPIError,
    CompletionHTTPError,
)


@pytest.mark.unit
class TestCompletionAdapterError:
    def test_message_attribute(self) -> None:
        err = CompletionAdapterError("something failed")
        assert err.message == "something failed", (
            "CompletionAdapterError.message must hold the constructor string verbatim."
        )

    def test_str_representation(self) -> None:
        err = CompletionAdapterError("something failed")
        assert str(err) == "something failed", (
            "__str__ must return the message string so formatted error output is readable."
        )

    def test_is_exception_subclass(self) -> None:
        assert isinstance(CompletionAdapterError("x"), Exception), (
            "CompletionAdapterError must be catchable as a plain Exception."
        )

    def test_reduce_reconstructs_correctly(self) -> None:
        err = CompletionAdapterError("msg")
        cls, args = err.__reduce__()
        assert cls is CompletionAdapterError, "__reduce__ must return the concrete class for pickling."
        assert args == ("msg",), "__reduce__ args must match the constructor signature."


@pytest.mark.unit
class TestCompletionAPIError:
    def test_model_name_attribute(self) -> None:
        err = CompletionAPIError("gpt-4", "connection refused")
        assert err.model_name == "gpt-4", "CompletionAPIError.model_name must hold the model string for diagnostics."

    def test_message_inherited(self) -> None:
        err = CompletionAPIError("gpt-4", "connection refused")
        assert str(err) == "connection refused", "str() must delegate to the inherited message, not the model_name."

    def test_isinstance_chain(self) -> None:
        err = CompletionAPIError("gpt-4", "connection refused")
        assert isinstance(err, CompletionAdapterError), (
            "CompletionAPIError must be catchable as CompletionAdapterError."
        )
        assert isinstance(err, Exception), "CompletionAPIError must be catchable as a plain Exception."

    def test_reduce_reconstructs_correctly(self) -> None:
        err = CompletionAPIError("gpt-4", "connection refused")
        cls, args = err.__reduce__()
        assert cls is CompletionAPIError
        assert args == ("gpt-4", "connection refused"), (
            "__reduce__ args must match (model_name, message) so pickling round-trips correctly."
        )


@pytest.mark.unit
class TestCompletionHTTPError:
    def test_status_code_attribute(self) -> None:
        err = CompletionHTTPError(429, "gpt-4", {"error": "rate_limit"})
        assert err.status_code == 429

    def test_body_attribute(self) -> None:
        body = {"error": "rate_limit"}
        err = CompletionHTTPError(429, "gpt-4", body)
        assert err.body is body

    def test_body_defaults_to_none(self) -> None:
        err = CompletionHTTPError(404, "gpt-4")
        assert err.body is None, "body should default to None when the vendor provides no response body."

    def test_message_contains_all_fields(self) -> None:
        err = CompletionHTTPError(503, "claude-3", "overloaded")
        assert "503" in str(err), "message must embed status_code for log readability"
        assert "claude-3" in str(err), "message must embed model_name for log readability"
        assert "overloaded" in str(err), "message must embed body for log readability"

    def test_isinstance_chain(self) -> None:
        err = CompletionHTTPError(500, "gpt-4")
        assert isinstance(err, CompletionHTTPError)
        assert isinstance(err, CompletionAPIError), "CompletionHTTPError must be catchable as CompletionAPIError."
        assert isinstance(err, CompletionAdapterError), (
            "CompletionHTTPError must be catchable as CompletionAdapterError."
        )

    def test_reduce_reconstructs_correctly(self) -> None:
        body = {"detail": "server error"}
        err = CompletionHTTPError(500, "gpt-4", body)
        cls, args = err.__reduce__()
        assert cls is CompletionHTTPError
        assert args == (500, "gpt-4", body), "__reduce__ args must match (status_code, model_name, body) for pickling."
