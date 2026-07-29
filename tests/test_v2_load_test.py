from __future__ import annotations

import json

from scripts.load_test import parse_sse_data


def test_parse_sse_data_handles_content_and_done() -> None:
    payload = {"choices": [{"delta": {"content": "hello"}}]}
    assert parse_sse_data(f"data: {json.dumps(payload)}") == payload
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data(": keep-alive") is None
