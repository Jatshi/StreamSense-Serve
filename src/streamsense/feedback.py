from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .evidence_agent import EvidenceAgentRequest, EvidenceAgentResponse

FeedbackRating = Literal["positive", "negative", "correction"]


class FeedbackSubmission(BaseModel):
    request: EvidenceAgentRequest
    response: EvidenceAgentResponse
    rating: FeedbackRating
    corrected_answer: str | None = Field(default=None, min_length=1, max_length=8_000)
    corrected_response: EvidenceAgentResponse | None = None
    consent_for_training: bool = False
    source_license: str | None = Field(default=None, min_length=2, max_length=256)
    reason_codes: list[str] = Field(default_factory=list, max_length=20)
    notes: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def validate_request_and_correction(self) -> FeedbackSubmission:
        if self.request.request_id != self.response.request_id:
            raise ValueError("request and response request_id values must match")
        if (
            self.corrected_response
            and self.request.request_id != self.corrected_response.request_id
        ):
            raise ValueError("corrected_response request_id must match the request")
        if (
            self.rating == "correction"
            and not self.corrected_answer
            and not self.corrected_response
        ):
            raise ValueError("correction feedback requires a corrected answer or response")
        if self.consent_for_training and not self.source_license:
            raise ValueError("source_license is required when consent_for_training is true")
        return self


class StoredFeedback(BaseModel):
    feedback_id: str
    content_hash: str
    duplicate: bool = False
    created_at: datetime
    submission: FeedbackSubmission


class ExportSummary(BaseModel):
    schema_version: Literal[2] = 2
    output_directory: str
    sft_path: str
    dpo_path: str
    bridge_path: str
    raw_path: str
    manifest_path: str
    sft_examples: int
    dpo_examples: int
    bridge_examples: int
    source_records: int
    eligible_records: int
    skipped_no_consent: int
    output_sha256: dict[str, str]


class FeedbackStore:
    """Append-oriented, deduplicated hard-case store backed by SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if os.name != "nt":
            self.database_path.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback(request_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_rating_created "
                "ON feedback(rating, created_at)"
            )

    @staticmethod
    def _content_hash(submission: FeedbackSubmission) -> str:
        canonical = submission.model_dump_json(exclude_none=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def add(self, submission: FeedbackSubmission) -> StoredFeedback:
        content_hash = self._content_hash(submission)
        now = datetime.now(timezone.utc)
        feedback_id = f"fb_{uuid4().hex}"
        payload = submission.model_dump_json()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT feedback_id, payload, created_at FROM feedback WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing:
                return StoredFeedback(
                    feedback_id=existing["feedback_id"],
                    content_hash=content_hash,
                    duplicate=True,
                    created_at=datetime.fromisoformat(existing["created_at"]),
                    submission=FeedbackSubmission.model_validate_json(existing["payload"]),
                )
            connection.execute(
                """
                INSERT INTO feedback(
                    feedback_id, content_hash, request_id, rating, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    content_hash,
                    submission.request.request_id,
                    submission.rating,
                    payload,
                    now.isoformat(),
                ),
            )
        return StoredFeedback(
            feedback_id=feedback_id,
            content_hash=content_hash,
            created_at=now,
            submission=submission,
        )

    def list(self, *, limit: int = 1_000) -> list[StoredFeedback]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_id, content_hash, payload, created_at "
                "FROM feedback ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            StoredFeedback(
                feedback_id=row["feedback_id"],
                content_hash=row["content_hash"],
                created_at=datetime.fromisoformat(row["created_at"]),
                submission=FeedbackSubmission.model_validate_json(row["payload"]),
            )
            for row in rows
        ]

    def export_training_data(self, output_directory: str | Path) -> ExportSummary:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        records = self.list(limit=10_000)
        sft_examples: list[dict[str, object]] = []
        dpo_examples: list[dict[str, object]] = []
        bridge_examples: list[dict[str, object]] = []
        raw_examples: list[dict[str, object]] = []
        eligible_records: list[StoredFeedback] = []
        for record in records:
            submission = record.submission
            if not submission.consent_for_training:
                continue
            if not submission.source_license:
                # Older records may predate the validation rule; fail closed during export.
                continue
            corrected = self._corrected_content(submission)
            if corrected is None:
                continue
            eligible_records.append(record)
            prompt = self._training_prompt(submission.request)
            source = {
                "schema_version": 2,
                "feedback_id": record.feedback_id,
                "request_id": submission.request.request_id,
                "content_hash": record.content_hash,
                "source_license": submission.source_license,
                "created_at": record.created_at.isoformat(),
            }
            sft_examples.append(
                {
                    "schema_version": 2,
                    "messages": [
                        {"role": "system", "content": "Answer only from cited evidence."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": corrected},
                    ],
                    "metadata": source,
                }
            )
            rejected = (
                self._response_content(submission.response)
                if submission.corrected_response is not None
                else submission.response.answer
            )
            if corrected.strip() != rejected.strip():
                dpo_examples.append(
                    {
                        "schema_version": 2,
                        "prompt": prompt,
                        "chosen": corrected,
                        "rejected": rejected,
                        "metadata": source,
                    }
                )
            if submission.corrected_response is not None:
                bridge_examples.append(
                    {
                        "schema_version": 2,
                        "request": submission.request.model_dump(mode="json"),
                        "target": self._response_payload(submission.corrected_response),
                        "metadata": source,
                    }
                )
            raw_examples.append(
                {
                    "schema_version": 2,
                    "feedback_id": record.feedback_id,
                    "content_hash": record.content_hash,
                    "source_license": submission.source_license,
                    "created_at": record.created_at.isoformat(),
                    "submission": submission.model_dump(mode="json"),
                }
            )

        sft_path = output_directory / "sft_candidates.jsonl"
        dpo_path = output_directory / "dpo_candidates.jsonl"
        bridge_path = output_directory / "evidenceagent_bridge.jsonl"
        raw_path = output_directory / "consented_feedback_raw.jsonl"
        manifest_path = output_directory / "export_manifest.json"
        self._write_jsonl_atomic(sft_path, sft_examples)
        self._write_jsonl_atomic(dpo_path, dpo_examples)
        self._write_jsonl_atomic(bridge_path, bridge_examples)
        self._write_jsonl_atomic(raw_path, raw_examples)
        output_paths = {
            "sft": sft_path,
            "dpo": dpo_path,
            "bridge": bridge_path,
            "raw": raw_path,
        }
        output_sha256 = {name: self._file_sha256(path) for name, path in output_paths.items()}
        source_hashes = sorted(record.content_hash for record in eligible_records)
        source_manifest_sha256 = hashlib.sha256(
            "\n".join(source_hashes).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "store_schema": "streamsense-feedback-v1",
                "eligible_records": len(eligible_records),
                "content_hashes": source_hashes,
                "source_manifest_sha256": source_manifest_sha256,
            },
            "outputs": {
                name: {
                    "file": path.name,
                    "sha256": output_sha256[name],
                    "examples": {
                        "sft": len(sft_examples),
                        "dpo": len(dpo_examples),
                        "bridge": len(bridge_examples),
                        "raw": len(raw_examples),
                    }[name],
                }
                for name, path in output_paths.items()
            },
        }
        self._write_json_atomic(manifest_path, manifest)
        return ExportSummary(
            output_directory=str(output_directory.resolve()),
            sft_path=str(sft_path.resolve()),
            dpo_path=str(dpo_path.resolve()),
            bridge_path=str(bridge_path.resolve()),
            raw_path=str(raw_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
            sft_examples=len(sft_examples),
            dpo_examples=len(dpo_examples),
            bridge_examples=len(bridge_examples),
            source_records=len(records),
            eligible_records=len(eligible_records),
            skipped_no_consent=sum(
                not record.submission.consent_for_training for record in records
            ),
            output_sha256=output_sha256,
        )

    @staticmethod
    def _training_prompt(request: EvidenceAgentRequest) -> str:
        evidence = [item.model_dump(mode="json") for item in request.evidence]
        return json.dumps(
            {"question": request.question, "evidence": evidence},
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def _corrected_content(cls, submission: FeedbackSubmission) -> str | None:
        if submission.corrected_response is not None:
            return json.dumps(
                cls._response_payload(submission.corrected_response),
                ensure_ascii=False,
                sort_keys=True,
            )
        return submission.corrected_answer

    @staticmethod
    def _response_payload(response: EvidenceAgentResponse) -> dict[str, object]:
        return {
            "state": response.state,
            "answer": response.answer,
            "citations": [citation.model_dump(mode="json") for citation in response.citations],
            "confidence": response.confidence,
            "missing_evidence": response.missing_evidence,
        }

    @classmethod
    def _response_content(cls, response: EvidenceAgentResponse) -> str:
        return json.dumps(
            cls._response_payload(response),
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
