"""Shared wall-clock and monotonic timing for GUI worker lifecycles."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time


SHORT_STEP_REFERENCE_SECONDS = {"freeze": 60.0}


def summarize_batch_outcomes(outcomes: object) -> tuple[str, dict[str, int]]:
    """Return an aggregate worker state and safe per-status counts."""
    counts = {"completed": 0, "failed": 0, "cancelled": 0, "unknown": 0}
    if not isinstance(outcomes, list):
        counts["unknown"] = 1
        return "failed", counts
    for outcome in outcomes:
        status = outcome.get("status") if isinstance(outcome, dict) else None
        if status in {"completed", "failed", "cancelled"}:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    if not outcomes or counts["failed"] or counts["unknown"]:
        return "failed", counts
    if counts["cancelled"]:
        return "cancelled", counts
    if counts["completed"] == len(outcomes):
        return "success", counts
    return "failed", counts


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分 {seconds_part} 秒"
    if minutes:
        return f"{minutes} 分 {seconds_part} 秒"
    return f"{seconds_part} 秒"


@dataclass
class RunTiming:
    """Track display timestamps separately from elapsed-time measurement."""

    action: str
    reference_seconds: float | None = None
    started_local: datetime | None = None
    started_monotonic: float | None = None
    ended_local: datetime | None = None
    ended_monotonic: float | None = None
    completed_work: int = 0
    total_work: int | None = None

    def start(self) -> None:
        self.started_local = datetime.now().astimezone()
        self.started_monotonic = time.monotonic()
        self.ended_local = None
        self.ended_monotonic = None

    def update_progress(self, completed: int, total: int | None) -> None:
        if type(completed) is int and completed >= 0:
            self.completed_work = completed
        if type(total) is int and total > 0:
            self.total_work = total

    @property
    def elapsed_seconds(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        end = self.ended_monotonic
        return max(0.0, (end if end is not None else time.monotonic()) - self.started_monotonic)

    @property
    def reference_exceeded(self) -> bool:
        return (
            self.reference_seconds is not None
            and self.elapsed_seconds > self.reference_seconds
            and not (self.completed_work > 0 and self.total_work is not None)
        )

    @property
    def remaining_seconds(self) -> float | None:
        if self.completed_work > 0 and self.total_work is not None:
            remaining = max(0, self.total_work - self.completed_work)
            return max(0.0, self.elapsed_seconds / self.completed_work * remaining)
        if self.reference_seconds is not None:
            if self.reference_exceeded:
                return None
            return max(0.0, self.reference_seconds - self.elapsed_seconds)
        return None

    @property
    def expected_finish_local(self) -> datetime | None:
        remaining = self.remaining_seconds
        if remaining is None:
            return None
        if self.completed_work > 0 and self.total_work is not None:
            return datetime.now().astimezone() + timedelta(seconds=remaining)
        if self.started_local is None or self.reference_seconds is None:
            return None
        return self.started_local + timedelta(seconds=self.reference_seconds)

    def finish(self) -> tuple[datetime, float]:
        if self.ended_monotonic is None:
            self.ended_monotonic = time.monotonic()
            self.ended_local = datetime.now().astimezone()
        return self.ended_local, self.elapsed_seconds
