"""Deterministic fault schedules for reproducible serving chaos tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FaultKind(str, Enum):
    TIMEOUT = "timeout"
    OOM = "oom"
    PROCESS_EXIT = "process_exit"
    MALFORMED_RESPONSE = "malformed_response"
    TRACE_EXPORT_FAILURE = "trace_export_failure"


@dataclass(frozen=True, slots=True)
class FaultSchedule:
    every_n_requests: int
    fault: FaultKind
    start_at: int = 1

    def __post_init__(self) -> None:
        if self.every_n_requests <= 0:
            raise ValueError("every_n_requests must be positive")
        if self.start_at <= 0:
            raise ValueError("start_at must be positive")

    def fault_for(self, request_number: int) -> FaultKind | None:
        if request_number < self.start_at:
            return None
        offset = request_number - self.start_at + 1
        return self.fault if offset % self.every_n_requests == 0 else None
