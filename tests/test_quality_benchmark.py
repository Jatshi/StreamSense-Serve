from __future__ import annotations

from scripts.quality_benchmark import score_output


def test_score_output_checks_expected_forbidden_and_json() -> None:
    assert score_output(
        {
            "expected_contains": ["answered"],
            "forbidden_contains": ["abstained"],
        },
        "answered",
    )["passed"]
    assert not score_output(
        {
            "expected_contains": ["answered"],
            "forbidden_contains": ["abstained"],
        },
        "answered, not abstained",
    )["passed"]
    result = score_output(
        {
            "expected_json": {"status": "answered", "confidence": 0.9},
            "forbidden_contains": ["```"],
        },
        '{"status":"answered","confidence":0.9}',
    )
    assert result["passed"]
    assert result["json_pass"] is True
