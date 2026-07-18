from __future__ import annotations

import json
import sys

from streamsense.cli import main
from tests.test_media import write_test_wave


def test_route_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streamsense",
            "route",
            "--risk",
            "0.9",
            "--uncertainty",
            "0.1",
            "--conflict",
            "0.0",
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["route"] == "vlm_escalated"


def test_analyze_cli(monkeypatch, capsys, tmp_path) -> None:
    media = tmp_path / "tone.wav"
    write_test_wave(media)
    database = tmp_path / "events.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streamsense",
            "analyze",
            str(media),
            "--stream-id",
            "cli-demo",
            "--database",
            str(database),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ],
    )
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["events_created"] >= 1
    assert database.is_file()
