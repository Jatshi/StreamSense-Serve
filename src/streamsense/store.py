from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .schema import EventRecord


class EventStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_stream_time ON events(stream_id, start_ms)"
            )

    def upsert(self, event: EventRecord) -> None:
        payload = event.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(
                    event_id, stream_id, event_type, start_ms, summary, payload, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    stream_id=excluded.stream_id,
                    event_type=excluded.event_type,
                    start_ms=excluded.start_ms,
                    summary=excluded.summary,
                    payload=excluded.payload,
                    created_at=excluded.created_at
                """,
                (
                    event.event_id,
                    event.stream_id,
                    event.event_type,
                    event.start_ms,
                    event.summary,
                    payload,
                    event.created_at.isoformat(),
                ),
            )

    def get(self, event_id: str) -> EventRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return EventRecord.model_validate_json(row["payload"]) if row else None

    def list(self, stream_id: str | None = None, limit: int = 100) -> list[EventRecord]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as connection:
            if stream_id is None:
                rows = connection.execute(
                    "SELECT payload FROM events ORDER BY start_ms ASC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload FROM events WHERE stream_id = ? ORDER BY start_ms ASC LIMIT ?",
                    (stream_id, limit),
                ).fetchall()
        return [EventRecord.model_validate_json(row["payload"]) for row in rows]

    def search(
        self, query: str, *, stream_id: str | None = None, limit: int = 5
    ) -> list[EventRecord]:
        if not query.strip():
            return []
        pattern = f"%{query.strip()}%"
        with self._connect() as connection:
            if stream_id is None:
                rows = connection.execute(
                    "SELECT payload FROM events WHERE summary LIKE ? OR payload LIKE ? "
                    "ORDER BY start_ms ASC LIMIT ?",
                    (pattern, pattern, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload FROM events WHERE stream_id = ? "
                    "AND (summary LIKE ? OR payload LIKE ?) ORDER BY start_ms ASC LIMIT ?",
                    (stream_id, pattern, pattern, limit),
                ).fetchall()
        return [EventRecord.model_validate_json(row["payload"]) for row in rows]
