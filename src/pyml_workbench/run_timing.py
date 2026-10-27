"""Shared wall-clock and monotonic timing for GUI worker lifecycles."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
import math
import time


SHORT_STEP_REFERENCE_SECONDS = {"freeze": 60.0}
DEFAULT_ACTION_ESTIMATES_SECONDS = {
    "train": 60.0, "wizard_batch": 60.0, "freeze": 60.0,
    "test": 30.0, "export": 30.0, "restore": 15.0,
    "inspect": 15.0, "inference": 30.0,
    "batch-run-request": 60.0, "batch-run": 60.0,
    "batch-freeze": 15.0, "batch-finalize": 60.0,
    "batch-export-exploration": 30.0, "batch-control": 5.0,
}
ACTION_STAGE_LABELS = {
    "train": "正在准备训练与验证", "wizard_batch": "正在预检并建立搜索队列",
    "freeze": "正在冻结模型设置", "test": "正在执行最终测试",
    "export": "正在导出结果", "restore": "正在恢复会话",
    "inspect": "正在读取模型信息", "inference": "正在执行新数据推理",
    "batch-run-request": "正在预检并建立搜索队列", "batch-run": "正在恢复验证搜索",
    "batch-freeze": "正在冻结验证优胜配置", "batch-finalize": "正在准备重拟合和最终测试",
    "batch-export-exploration": "正在导出训练探索结果", "batch-control": "正在保存队列控制请求",
}


def _rolling_remaining(estimate: float, elapsed: float) -> float:
    """Keep unfinished work visible after a reference estimate is exceeded."""
    if elapsed < estimate:
        return max(1.0, estimate - elapsed)
    return max(1.0, estimate * .25, elapsed * .25)


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
    stage_label: str = ""
    _stage_started_elapsed: float = field(default=0.0, init=False)
    _paused_at: float | None = field(default=None, init=False)
    _paused_seconds: float = field(default=0.0, init=False)
    _progress_elapsed: float = field(default=0.0, init=False)
    _external_remaining: float | None = field(default=None, init=False)
    _external_anchor: float = field(default=0.0, init=False)
    _external_exceeded: bool = field(default=False, init=False)
    _estimate_basis: str = field(default="初始估计", init=False)

    def __post_init__(self) -> None:
        if self.reference_seconds is None:
            self.reference_seconds = DEFAULT_ACTION_ESTIMATES_SECONDS.get(self.action, 30.0)
        if not math.isfinite(self.reference_seconds) or self.reference_seconds <= 0:
            raise ValueError("Timing reference must be positive and finite")
        if not self.stage_label:
            self.stage_label = ACTION_STAGE_LABELS.get(self.action, "正在执行后台任务")

    def start(self, stage_label: str | None = None, reference_seconds: float | None = None) -> None:
        self.started_local = datetime.now().astimezone()
        self.started_monotonic = time.monotonic()
        self.ended_local = None
        self.ended_monotonic = None
        self._paused_at = None
        self._paused_seconds = 0.0
        self._stage_started_elapsed = 0.0
        self._progress_elapsed = 0.0
        self._external_remaining = None
        if stage_label is not None:
            self.stage_label = stage_label
        if reference_seconds is not None:
            self.reference_seconds = reference_seconds

    def set_stage(self, label: str, reference_seconds: float | None = None) -> None:
        if self.ended_monotonic is not None:
            return
        if label == self.stage_label and reference_seconds is None:
            return
        self.stage_label = label
        self._stage_started_elapsed = self.elapsed_seconds
        self.reference_seconds = reference_seconds if reference_seconds is not None else DEFAULT_ACTION_ESTIMATES_SECONDS.get(self.action, 30.0)
        self.completed_work, self.total_work = 0, None
        self._progress_elapsed = 0.0
        self._external_remaining = None
        self._estimate_basis = "初始估计"

    @property
    def is_paused(self) -> bool:
        return self._paused_at is not None

    def pause(self) -> None:
        if self.started_monotonic is not None and not self.is_paused and self.ended_monotonic is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_seconds += max(0.0, time.monotonic() - self._paused_at)
            self._paused_at = None

    def set_remaining_estimate(self, seconds: float, basis: str = "依据已完成拟合实测", exceeded: bool = False) -> None:
        self._external_remaining = max(1.0, seconds)
        self._external_anchor = self.elapsed_seconds
        self._external_exceeded = exceeded
        self._estimate_basis = basis

    def update_progress(self, completed: int, total: int | None) -> None:
        if type(completed) is int and completed >= 0:
            if completed > self.completed_work:
                self._progress_elapsed = self.stage_elapsed_seconds
            self.completed_work = completed
        if type(total) is int and total > 0:
            self.total_work = total

    @property
    def elapsed_seconds(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        end = self.ended_monotonic
        if end is None:
            end = self._paused_at if self._paused_at is not None else time.monotonic()
        return max(0.0, end - self.started_monotonic - self._paused_seconds)

    @property
    def stage_elapsed_seconds(self) -> float:
        return max(0.0, self.elapsed_seconds - self._stage_started_elapsed)

    @property
    def estimate_basis(self) -> str:
        if self._external_remaining is not None:
            return self._estimate_basis
        if self.completed_work > 0 and self.total_work is not None:
            return "依据已完成工作实测"
        return "初始估计"

    @property
    def reference_exceeded(self) -> bool:
        if self.ended_monotonic is not None:
            return False
        if self._external_remaining is not None:
            return self._external_exceeded or self.elapsed_seconds - self._external_anchor > self._external_remaining
        if self.completed_work > 0 and self.total_work is not None:
            return self.completed_work < self.total_work and self.stage_elapsed_seconds - self._progress_elapsed > self._progress_elapsed / self.completed_work
        return self.stage_elapsed_seconds > self.reference_seconds

    @property
    def remaining_seconds(self) -> float:
        if self.ended_monotonic is not None:
            return 0.0
        if self._external_remaining is not None:
            return _rolling_remaining(self._external_remaining, self.elapsed_seconds - self._external_anchor)
        if self.completed_work > 0 and self.total_work is not None:
            remaining = max(0, self.total_work - self.completed_work)
            if remaining == 0:
                return _rolling_remaining(5.0, self.stage_elapsed_seconds - self._progress_elapsed)
            unit = max(1e-6, self._progress_elapsed / self.completed_work)
            current = self.stage_elapsed_seconds - self._progress_elapsed
            return unit * (remaining - 1) + _rolling_remaining(unit, current)
        return _rolling_remaining(self.reference_seconds, self.stage_elapsed_seconds)

    @property
    def expected_finish_local(self) -> datetime | None:
        remaining = self.remaining_seconds
        if self.ended_local is not None:
            return self.ended_local
        if self.is_paused:
            return None
        return datetime.now().astimezone() + timedelta(seconds=remaining)

    def finish(self) -> tuple[datetime, float]:
        if self.ended_monotonic is None:
            self.resume()
            self.ended_monotonic = time.monotonic()
            self.ended_local = datetime.now().astimezone()
        return self.ended_local, self.elapsed_seconds


@dataclass
class _SearchJobTiming:
    label: str
    limit: int
    max_proposals: int
    timeout_seconds: float
    status: str = "queued"
    actual_fit_count: int = 0
    proposal_count: int = 0
    completed_fit_count: int = 0
    fit_duration_count: int = 0
    fit_duration_total: float = 0.0
    elapsed_seconds: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)
    fit_started_at: float | None = None
    active_token: object = None
    fit_paused_at: float | None = None
    seen_fits: set = field(default_factory=set)
    finished: bool = False
    fits_per_candidate: int = 1
    active_fit_count: int = 0


class SearchTimingEstimator:
    """Estimate remaining search work using per-job observations and FIFO lanes.

    Configuration and worker messages are input boundaries. Timer refreshes use
    only these in-memory snapshots; they never read datasets or trial histories.
    """

    def __init__(self, parallel_workers: int = 1, default_fit_seconds: float = 60.0):
        self.parallel_workers = max(1, int(parallel_workers))
        self.default_fit_seconds = float(default_fit_seconds)
        self.jobs: dict[str, _SearchJobTiming] = {}

    def configure_job(self, job_id: str, spec, model_label: str = "", elapsed_seconds: float = 0,
                      actual_fit_count: int = 0, proposal_count: int = 0,
                      completed_fit_count: int = 0, fit_duration_count: int = 0,
                      fit_duration_total: float = 0, status: str = "queued",
                      fits_per_candidate: int = 1) -> None:
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        if not isinstance(spec, dict):
            raise ValueError("Search timing needs a search specification")
        limit = int(spec.get("max_fits", spec.get("max_actual_fits", 1)))
        proposals = int(spec.get("max_proposals", max(1, limit)))
        fits_per_candidate = max(1, int(fits_per_candidate))
        if spec.get("method") == "grid":
            fields = spec.get("space", {}).get("fields", {})
            grid_size = 1
            for item in fields.values():
                values = item.get("values")
                if not isinstance(values, list):
                    break
                grid_size *= len(values)
            else:
                limit = min(limit, grid_size * fits_per_candidate)
                proposals = min(proposals, grid_size)
        numbers = (elapsed_seconds, fit_duration_total)
        if limit < 0 or proposals <= 0 or any(not math.isfinite(x) or x < 0 for x in numbers):
            raise ValueError("Invalid search timing metadata")
        timeout = float(spec.get("timeout_seconds", 3600.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid search timing budget")
        job = self.jobs.get(job_id)
        if job is None:
            job = _SearchJobTiming(model_label or job_id, min(limit, proposals * fits_per_candidate), proposals, timeout)
            self.jobs[job_id] = job
        else:
            job.limit, job.max_proposals, job.timeout_seconds = min(limit, proposals * fits_per_candidate), proposals, timeout
            job.label = model_label or job.label
        job.elapsed_seconds = max(job.elapsed_seconds, float(elapsed_seconds))
        job.actual_fit_count = max(job.actual_fit_count, actual_fit_count)
        job.proposal_count = max(job.proposal_count, proposal_count)
        job.completed_fit_count = max(job.completed_fit_count, completed_fit_count)
        if fit_duration_count > job.fit_duration_count:
            job.fit_duration_count = fit_duration_count
            job.fit_duration_total = float(fit_duration_total)
        job.status = status
        job.fits_per_candidate = fits_per_candidate
        job.finished = status in {"completed", "failed", "cancelled"}
        job.updated_at = time.monotonic()

    def update_event(self, payload: dict) -> None:
        job_id = payload.get("job_id")
        data = payload.get("data") or {}
        name = payload.get("name") or payload.get("operation")
        if not job_id or not isinstance(data, dict):
            return
        if name == "search_started":
            history = data.get("resume_timing") or {}
            self.configure_job(job_id, data.get("spec") or {},
                               elapsed_seconds=data.get("elapsed_seconds", 0),
                               actual_fit_count=data.get("actual_fit_count", 0),
                               proposal_count=data.get("proposal_count", 0), status="running",
                               fits_per_candidate=data.get("fits_per_candidate", 1),
                               **{key: history.get(key, 0) for key in ("completed_fit_count", "fit_duration_count", "fit_duration_total")})
        job = self.jobs.get(job_id)
        if job is None or job.finished:
            return
        now = time.monotonic()
        # Worker protocol validation happens before dispatch to this boundary.
        if "actual_fit_count" in data:
            job.actual_fit_count = max(job.actual_fit_count, data["actual_fit_count"])
        if "proposal_count" in data:
            job.proposal_count = max(job.proposal_count, data["proposal_count"])
        if "elapsed_seconds" in data:
            job.elapsed_seconds = max(job.elapsed_seconds, data["elapsed_seconds"])
        job.updated_at = now
        if name == "fit_started":
            record = data.get("record") or {}
            token = record.get("trial_id") or payload.get("trial_id") or job.actual_fit_count
            if token not in job.seen_fits and job.fit_started_at is None:
                job.fit_started_at = now
                job.active_token = token
            elif data.get("fold") is not None and job.active_token == token:
                job.fit_started_at = now
            job.active_fit_count = int(record.get("fit_count") or 1)
            job.status = "running"
        elif name == "fit_completed":
            record = data.get("record") or {}
            token = record.get("trial_id") or payload.get("trial_id") or job.actual_fit_count
            if token not in job.seen_fits:
                job.seen_fits.add(token)
                fit_count = int(record.get("fit_count") or 1)
                job.completed_fit_count += fit_count
                duration = record.get("duration_seconds")
                if record.get("status") == "complete" and isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration > 0:
                    job.fit_duration_count += fit_count
                    job.fit_duration_total += duration
            if job.active_token == token:
                job.fit_started_at = None
                job.active_token = None
                job.active_fit_count = 0
        elif name == "search_paused":
            job.status = "paused"
            job.fit_paused_at = now
        elif name == "search_resumed":
            if job.fit_started_at is not None and job.fit_paused_at is not None:
                job.fit_started_at += now - job.fit_paused_at
            job.fit_paused_at = None
            job.status = "running"
        elif name == "search_finished":
            job.finished = True
            job.status = (data.get("result") or {}).get("status", "completed")
            job.fit_started_at = None

    @property
    def is_paused(self) -> bool:
        pending = [job for job in self.jobs.values() if not job.finished]
        return bool(pending) and any(job.status == "paused" for job in pending) and not any(job.status == "running" for job in pending)

    @property
    def estimate_basis(self) -> str:
        pending = [job for job in self.jobs.values() if not job.finished]
        count = sum(job.fit_duration_count for job in pending)
        if not count:
            return "初始估计（每次拟合约 1 分钟）"
        if any(not job.fit_duration_count for job in pending):
            return f"依据 {count} 次成功拟合实测，其余任务使用初始估计"
        return f"依据 {count} 次成功拟合实测"

    def _unit(self, job: _SearchJobTiming) -> float:
        return max(1e-6, job.fit_duration_total / job.fit_duration_count) if job.fit_duration_count else self.default_fit_seconds

    @property
    def reference_exceeded(self) -> bool:
        now = time.monotonic()
        return any(not job.finished and job.fit_started_at is not None and (job.fit_paused_at if job.fit_paused_at is not None else now) - job.fit_started_at > self._unit(job) for job in self.jobs.values())

    def _remaining(self, job: _SearchJobTiming, now: float) -> float:
        if job.finished:
            return 0.0
        unit = self._unit(job)
        pending_inside = max(0, job.fits_per_candidate - job.active_fit_count) if job.fit_started_at is not None else 0
        count = max(0, min(job.limit - job.actual_fit_count,
                           (job.max_proposals - job.proposal_count) * job.fits_per_candidate + pending_inside))
        inflight = job.fit_started_at is not None
        fit_elapsed = max(0.0, (job.fit_paused_at if job.fit_paused_at is not None else now) - job.fit_started_at) if inflight else 0.0
        current = _rolling_remaining(unit, fit_elapsed) if inflight else 0.0
        future = count * unit
        # A time budget is a soft safe-point limit, never an interrupt deadline.
        elapsed = job.elapsed_seconds + (max(0.0, now - job.updated_at) if job.status == "running" else 0.0)
        budget_left = max(0.0, job.timeout_seconds - elapsed)
        future = min(future, max(0.0, budget_left - current))
        return max(1.0, current + future)

    def remaining_seconds(self) -> float:
        if not self.jobs:
            return self.default_fit_seconds
        now = time.monotonic()
        pending = [job for job in self.jobs.values() if not job.finished]
        if not pending:
            return 1.0  # Worker still needs to persist and return its summary.
        # Running jobs already occupy lanes. Fill queued work into the first
        # available lane, preserving FIFO and the slowest active task's tail.
        active = [job for job in pending if job.status in {"running", "paused"}]
        queued = [job for job in pending if job.status not in {"running", "paused"}]
        lanes = [0.0] * max(self.parallel_workers, len(active))
        for index, job in enumerate(active):
            lanes[index] = self._remaining(job, now)
        heapq.heapify(lanes)
        for job in queued:
            free = heapq.heappop(lanes)
            heapq.heappush(lanes, free + self._remaining(job, now))
        return max(1.0, max(lanes))
