"""The three module services report failures in three different shapes.

parse_upstream_error must collapse all of them, plus non-JSON bodies, into one
predictable tuple.
"""
import httpx

from app.core.errors import parse_upstream_error


def response(status_code: int, *, json_body=None, text: str = "") -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code, text=text)


def test_reasoning_service_envelope():
    body = {
        "error": True,
        "error_code": "EMPTY_SCHEME",
        "message": "Marking scheme has no steps.",
        "details": {"hint": "check the image"},
        "status_code": 400,
    }
    code, message, details = parse_upstream_error(response(400, json_body=body))
    assert code == "EMPTY_SCHEME"
    assert message == "Marking scheme has no steps."
    assert details == {"hint": "check the image"}


def test_fastapi_http_exception_detail_string():
    """ocr-service and feedback-service both raise plain HTTPExceptions."""
    code, message, details = parse_upstream_error(
        response(503, json_body={"detail": "OPENAI_API_KEY is not set."})
    )
    assert code is None
    assert message == "OPENAI_API_KEY is not set."
    assert details == {}


def test_fastapi_validation_error_detail_list():
    body = {
        "detail": [
            {"loc": ["body", "student_steps", 0, "expression"], "msg": "Field required", "type": "missing"},
            {"loc": ["body", "assigned_marks"], "msg": "Input should be a valid number", "type": "float_type"},
        ]
    }
    code, message, details = parse_upstream_error(response(422, json_body=body))
    assert code == "VALIDATION_ERROR"
    assert "body.student_steps.0.expression: Field required" in message
    assert "body.assigned_marks: Input should be a valid number" in message
    assert len(details["errors"]) == 2


def test_non_json_body_falls_back_to_text():
    code, message, details = parse_upstream_error(
        response(502, text="<html><body>Bad Gateway</body></html>")
    )
    assert code is None
    assert "Bad Gateway" in message
    assert details == {}


def test_empty_body_reports_the_status_code():
    code, message, _ = parse_upstream_error(response(500, text=""))
    assert code is None
    assert message == "HTTP 500"


def test_long_bodies_are_truncated():
    _, message, _ = parse_upstream_error(response(500, text="x" * 5000))
    assert len(message) <= 500


def test_unrecognised_json_shape_is_stringified():
    _, message, _ = parse_upstream_error(response(500, json_body={"oops": "unexpected"}))
    assert "oops" in message
