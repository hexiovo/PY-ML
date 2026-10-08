"""Trusted adapters from loaded data and owned experiment caches to PlotSource."""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd


class PlotCacheError(ValueError):
    """Raised when a plot source is missing, invalid, or not owned by this app."""


_PLOT_CACHE_MANIFEST_KIND = "pyml-workbench-plot-cache-manifest"
_PLOT_CACHE_MANIFEST_DOMAIN = "pyml-workbench.plot-cache-manifest"


def session_plot_cache_manifest_path(session_path: str | Path) -> Path:
    path = Path(session_path).expanduser().resolve()
    return path.with_name(path.name + ".plot-cache-manifest.json")


def build_session_plot_cache_manifest(session) -> dict[str, Any]:
    """Fingerprint the fitted-session data used to produce cached result plots.

    The manifest is stored beside the owned session in the same internal cache
    envelope. It detects cache-content drift; it is not a signature against an
    actor who can rewrite both the session and its manifest.
    """
    from .experiment import ExperimentSession, _table_digest

    if not isinstance(session, ExperimentSession):
        raise PlotCacheError("无法为不兼容的 session 创建绘图缓存 manifest。")
    try:
        if not isinstance(session.splits, Mapping) or set(session.splits) != {
            "train", "validation", "test"
        }:
            raise ValueError("session split 集合不完整")
        split_positions = {
            name: [int(value) for value in np.asarray(session.splits[name], dtype=np.int64)]
            for name in ("train", "validation", "test")
        }
        sequence_plan = getattr(session, "sequence_plan", None)
        if sequence_plan is not None:
            try:
                sequence_plan.verify(config=session.config)
            except Exception as exc:
                raise PlotCacheError(
                    f"session sequence mapping 与绘图缓存 manifest 不匹配：{exc}"
                ) from exc
            sequence_mapping = _sequence_plot_cache_payload(sequence_plan)
        else:
            sequence_mapping = None
        result = getattr(session, "result", None)
        result_rows = None
        if result is not None:
            result_frame = getattr(result, "results", None)
            if not isinstance(result_frame, pd.DataFrame):
                raise ValueError("session result rows 格式无效")
            result_rows = _plot_cache_table_payload(result_frame)
        training_rows = _plot_cache_records_payload(getattr(session, "rows", []))
        plot_scores = getattr(session, "plot_scores", {})
        plot_explanations = getattr(session, "plot_explanations", {})
        if not isinstance(plot_scores, Mapping) or not isinstance(plot_explanations, Mapping):
            raise ValueError("session score/explanation cache metadata must be mappings")
        digests = {
            "frozen_config": _plot_cache_domain_sha256(
                "frozen_config",
                {
                    "frozen_config": session.frozen_config,
                    "frozen_config_sha256": session.frozen_config_sha256,
                },
            ),
            "features": _plot_cache_domain_sha256(
                "features", {"table_sha256": _table_digest(session.features)}
            ),
            "target": _plot_cache_domain_sha256(
                "target", {"table_sha256": _table_digest(session.target)}
            ),
            "splits": _plot_cache_domain_sha256("splits", split_positions),
            "sequence_mapping": _plot_cache_domain_sha256(
                "sequence_mapping", sequence_mapping
            ),
            "result_rows": _plot_cache_domain_sha256(
                "result_rows",
                {"training_rows": training_rows, "final_results": result_rows},
            ),
            "scores": _plot_cache_domain_sha256("scores", dict(plot_scores)),
            "explanations": _plot_cache_domain_sha256("explanations", dict(plot_explanations)),
        }
        payload = {
            "schema_version": 2,
            "kind": _PLOT_CACHE_MANIFEST_KIND,
            "digests": digests,
        }
        return {
            **payload,
            "manifest_sha256": _plot_cache_domain_sha256("manifest", payload),
        }
    except PlotCacheError:
        raise
    except Exception as exc:
        raise PlotCacheError(f"无法核验 session 绘图缓存 manifest：{exc}") from exc


def _plot_cache_domain_sha256(domain: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prefix = f"{_PLOT_CACHE_MANIFEST_DOMAIN}\0v1\0{domain}\0".encode("utf-8")
    return hashlib.sha256(prefix + encoded).hexdigest()


def _verify_session_plot_cache_manifest(envelope: Mapping[str, Any], session) -> int:
    stored = envelope.get("plot_cache_manifest")
    if not isinstance(stored, dict):
        raise PlotCacheError(
            "本次 session 缓存缺少绘图完整性 manifest；旧缓存不可用于结果绘图，请重新训练。"
        )
    payload = {
        "schema_version": stored.get("schema_version"),
        "kind": stored.get("kind"),
        "digests": stored.get("digests"),
    }
    if (
        payload["schema_version"] not in {1, 2}
        or payload["kind"] != _PLOT_CACHE_MANIFEST_KIND
        or not isinstance(payload["digests"], dict)
        or not _is_sha256(stored.get("manifest_sha256"))
        or _plot_cache_domain_sha256("manifest", payload) != stored.get("manifest_sha256")
    ):
        raise PlotCacheError("本次 session 绘图缓存 manifest 自身摘要无效。")

    expected = build_session_plot_cache_manifest(session)
    stored_digests = payload["digests"]
    expected_digests = expected["digests"]
    if stored_digests.get("frozen_config") != expected_digests["frozen_config"]:
        raise PlotCacheError("session 冻结配置摘要与绘图缓存 manifest 不匹配。")
    if any(
        stored_digests.get(name) != expected_digests[name]
        for name in ("features", "target")
    ):
        raise PlotCacheError("session 数据哈希与绘图缓存 manifest 不匹配。")
    if stored_digests.get("splits") != expected_digests["splits"]:
        raise PlotCacheError("session split 哈希与绘图缓存 manifest 不匹配。")
    if stored_digests.get("sequence_mapping") != expected_digests["sequence_mapping"]:
        raise PlotCacheError("session sequence mapping 与绘图缓存 manifest 不匹配。")
    if stored_digests.get("result_rows") != expected_digests["result_rows"]:
        raise PlotCacheError("session prediction/result rows 与绘图缓存 manifest 不匹配。")
    if payload["schema_version"] == 1:
        # Version 1 caches predate real estimator scores and native explanation
        # metadata. Their original digests still protect all existing plot data.
        if any(name in stored_digests for name in ("scores", "explanations")):
            raise PlotCacheError("旧版 session 绘图缓存含有未知的评分或解释摘要。")
    else:
        if (
            stored_digests.get("scores") != expected_digests["scores"]
            or stored_digests.get("explanations") != expected_digests["explanations"]
            or stored != expected
        ):
            raise PlotCacheError("session 评分/解释缓存与完整性 manifest 不匹配。")
    return payload["schema_version"]


def _plot_cache_table_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {"label": _plot_cache_value_payload(name), "dtype": str(dtype)}
            for name, dtype in zip(frame.columns, frame.dtypes)
        ],
        "index": [_plot_cache_value_payload(value) for value in frame.index.tolist()],
        "rows": [
            [_plot_cache_value_payload(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _plot_cache_records_payload(records: Any) -> list[Any]:
    if not isinstance(records, (list, tuple)):
        raise ValueError("session training rows 格式无效")
    if any(not isinstance(row, Mapping) for row in records):
        raise ValueError("session training row 必须是对象")
    return [
        {
            str(key): _plot_cache_value_payload(value)
            for key, value in sorted(row.items(), key=lambda pair: str(pair[0]))
        }
        for row in records
    ]


def _sequence_plot_cache_payload(plan: Any) -> dict[str, Any]:
    def positions(value):
        return None if value is None else np.asarray(value).tolist()

    partitions = {}
    for name, partition in plan.partitions.items():
        partitions[name] = {
            "row_positions": positions(partition.row_positions),
            "group_ids": _plot_cache_value_payload(partition.group_ids),
            "lengths": positions(partition.lengths),
            "window_source_positions": positions(partition.window_source_positions),
            "window_target_positions": positions(partition.window_target_positions),
        }
    return {
        "plan_manifest": plan.to_dict(),
        "features_sha256": _plot_cache_domain_sha256(
            "sequence_features", {"table_sha256": _plot_cache_table_digest(plan._features_frame)}
        ),
        "target_sha256": _plot_cache_domain_sha256(
            "sequence_target", {"table_sha256": _plot_cache_table_digest(plan._target_series)}
        ),
        "split_positions": {
            name: [int(value) for value in values]
            for name, values in plan.split_positions.items()
        },
        "group_keys_by_row": _plot_cache_value_payload(plan._group_keys_by_row),
        "group_values_by_row": _plot_cache_value_payload(plan._group_values_by_row),
        "order_values_ns": positions(plan._order_values_ns),
        "partitions": partitions,
    }


def _plot_cache_table_digest(frame: pd.DataFrame | pd.Series | None) -> str | None:
    from .experiment import _table_digest

    return _table_digest(frame)


def _plot_cache_value_payload(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": [
                [_plot_cache_value_payload(key), _plot_cache_value_payload(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "items": _plot_cache_value_payload(value.tolist()),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_plot_cache_value_payload(item) for item in value],
        }
    return _typed_value(_scalar(value, context="session result row", allow_tuple=True))


@dataclass(frozen=True)
class PlotSourceCollection:
    """Available immutable plot sources plus reasons a requested partition is absent."""

    sources: tuple[Any, ...]
    unavailable: tuple[tuple[str, str], ...] = ()

    def by_partition(self) -> dict[str, Any]:
        return {source.partition: source for source in self.sources}

    def unavailable_by_partition(self) -> dict[str, str]:
        return dict(self.unavailable)


def loaded_dataset_plot_source(dataset, *, source_sha256: str | None = None):
    """Copy every loaded row/column into a detached, typed EDA source."""
    from .plotting import PlotColumn, PlotProvenance, PlotSource

    frame = getattr(dataset, "frame", None)
    if not isinstance(frame, pd.DataFrame) or not len(frame.index) or not len(frame.columns):
        raise PlotCacheError("当前已加载数据没有可绘制的完整表格。")
    row_positions = tuple(range(len(frame.index)))
    columns = tuple(
        _plot_column(
            f"data.{index}",
            label,
            str(frame.dtypes.iloc[index]),
            frame.iloc[:, index].tolist(),
        )
        for index, label in enumerate(frame.columns)
    )
    data_sha256 = _frame_sha256(frame)
    payload_sha256 = _source_payload_sha256(row_positions, columns, (), ())
    path = str(getattr(dataset, "source_path", ""))
    provenance = PlotProvenance(
        source_sha256=source_sha256,
        data_sha256=data_sha256,
        cache_payload_sha256=payload_sha256,
    )
    return PlotSource(
        schema_version=1,
        source_kind="loaded_eda",
        source_id=path or "loaded-dataset",
        owner_kind="loaded_dataset",
        owner_id=path or "loaded-dataset",
        partition="all",
        row_positions=row_positions,
        columns=columns,
        outputs=(),
        sequence_map=(),
        provenance=provenance,
        partition_count=len(row_positions),
    )


def select_plot_source_columns(source, selected_column_ids):
    """Return a detached source view whose payload hash covers only chosen columns."""
    from dataclasses import replace
    from .plotting import PlotProvenance, PlotSource

    if not isinstance(source, PlotSource):
        raise TypeError("source must be a PlotSource")
    selected = set(selected_column_ids)
    columns = tuple(column for column in source.columns if column.column_id in selected)
    outputs = tuple(column for column in source.outputs if column.column_id in selected)
    provenance = replace(
        source.provenance,
        cache_payload_sha256=_source_payload_sha256(
            source.row_positions, columns, outputs, source.sequence_map,
            source.score_cache, source.explanation_cache,
        ),
    )
    return replace(source, columns=columns, outputs=outputs, provenance=provenance)


def load_single_session_plot_sources(
    session_path: str | Path,
    *,
    session_id: str,
    frozen_config_sha256: str,
) -> PlotSourceCollection:
    """Read the current GUI-owned session and freeze plot values before cleanup."""
    path = Path(session_path).expanduser().resolve()
    if not path.is_file():
        raise PlotCacheError("本次实验的 session 缓存不存在，无法恢复结果图。")
    try:
        envelope = joblib.load(path)
    except Exception as exc:
        raise PlotCacheError(f"无法读取本次实验 session 缓存：{exc}") from exc
    if (
        not isinstance(envelope, dict)
        or envelope.get("kind") != "pyml-workbench-internal-session"
        or envelope.get("schema_version") != 1
        or envelope.get("session_id") != session_id
        or Path(str(envelope.get("session_path", ""))).expanduser().resolve() != path
    ):
        raise PlotCacheError("本次实验 session 身份或路径与当前窗口不匹配。")
    session = envelope.get("session")
    if session is None:
        raise PlotCacheError("本次实验 session 没有已拟合的结果缓存。")
    plot_manifest_version = _verify_session_plot_cache_manifest(envelope, session)
    manifest = _validate_session(session, expected_config_sha256=frozen_config_sha256)
    if envelope.get("frozen_config_sha256") != frozen_config_sha256:
        raise PlotCacheError("本次实验 session 配置哈希不匹配。")

    sources: list[Any] = []
    unavailable: list[tuple[str, str]] = []
    for partition in ("train", "validation"):
        source = _session_partition_source(
            session,
            partition,
            source_kind="session_cache",
            owner_kind="session",
            owner_id=session_id,
            source_id=session_id,
            manifest=manifest,
            plot_manifest_version=plot_manifest_version,
        )
        if source is not None:
            sources.append(source)
        else:
            unavailable.append((partition, f"本次 session 没有可用的 {partition} 结果行。"))
    if _test_cache_is_successful(session, envelope_state=envelope.get("state")):
        source = _session_partition_source(
            session,
            "test",
            source_kind="session_cache",
            owner_kind="session",
            owner_id=session_id,
            source_id=session_id,
            manifest=manifest,
            plot_manifest_version=plot_manifest_version,
        )
        if source is not None:
            sources.append(source)
        else:
            unavailable.append(("test", "最终测试成功，但 session 中没有可用的测试结果行。"))
    else:
        unavailable.append(("test", "最终测试没有通过成功状态、一次性计数、模型能力与本次 session 所有权核验。"))
    return PlotSourceCollection(tuple(sources), tuple(unavailable))


def load_batch_plot_sources(history_path: str | Path, job_id: str) -> PlotSourceCollection:
    """Load only result rows owned by the selected, completed durable batch job."""
    from .batch import load_search_result
    from .history import HistoryStore

    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise PlotCacheError("所选批量任务已从历史中移除。")
    if job.get("job_id") != job_id or job.get("status") != "completed":
        raise PlotCacheError("只有当前所有者的已完成批量任务可以提供结果图。")
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    receipt = _json_object(job.get("snapshot_receipt_json"), "快照收据")
    from .experiment import _json_digest

    snapshot_sha256 = str(job.get("snapshot_sha256") or "")
    snapshot_json = job.get("snapshot_json")
    if (
        not _is_sha256(snapshot_sha256)
        or not isinstance(snapshot_json, str)
        or hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest() != snapshot_sha256
    ):
        raise PlotCacheError("批量历史中的 snapshot_sha256 与快照记录不匹配。")
    receipt_manifest_sha256 = str(receipt.get("manifest_sha256") or "")
    receipt_file_sha256 = str(receipt.get("file_sha256") or "")
    if not _is_sha256(receipt_manifest_sha256) or not _is_sha256(receipt_file_sha256):
        raise PlotCacheError("批量快照收据缺少有效的独立哈希。")
    if (
        receipt.get("relative_path") != "snapshot.joblib"
        or _json_digest(receipt.get("manifest")) != receipt_manifest_sha256
    ):
        raise PlotCacheError("批量快照收据路径或 manifest 哈希不匹配。")
    try:
        result = load_search_result(history_path, job_id)
    except Exception as exc:
        if "plot-cache manifest" not in str(exc).lower():
            raise
        raise PlotCacheError(f"无法核验所选批量任务的 winner session：{exc}") from exc
    if result is None or result.job_id != job_id or result.winner is None or result.winner_session is None:
        raise PlotCacheError("所选任务没有可验证的 winner 结果缓存。")
    winner_path = Path(str(result.winner.session_path or "")).expanduser().resolve()
    _assert_owned_path(winner_path, artifact_dir, "winner session")
    if not winner_path.is_file():
        raise PlotCacheError("winner session 文件缺失。")
    winner_manifest_path = session_plot_cache_manifest_path(winner_path)
    if not winner_manifest_path.is_file():
        raise PlotCacheError(
            "winner session 缺少绘图完整性 manifest；其 train/validation 结果图不可用。"
        )
    try:
        winner_plot_manifest = json.loads(winner_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlotCacheError(f"无法读取 winner session 绘图 manifest：{exc}") from exc
    winner_plot_manifest_version = _verify_session_plot_cache_manifest(
        {"plot_cache_manifest": winner_plot_manifest}, result.winner_session
    )
    if result.snapshot.to_dict() != receipt.get("manifest"):
        raise PlotCacheError("winner 快照与所选 job 的 owned snapshot 收据不匹配。")
    snapshot_file = artifact_dir / "snapshot.joblib"
    _assert_owned_path(snapshot_file.resolve(), artifact_dir, "owned snapshot")
    if not snapshot_file.is_file() or _file_sha256(snapshot_file) != receipt_file_sha256:
        raise PlotCacheError("批量 owned snapshot 文件哈希与收据不匹配。")

    winner = result.winner
    if winner.status != "complete" or winner.test_evaluation_count != 0:
        raise PlotCacheError("批量 winner 不是未触及测试集的成功搜索缓存。")
    winner_session = result.winner_session
    winner_hash = str(winner.config_sha256 or "")
    _validate_session(winner_session, snapshot=result.snapshot, expected_config_sha256=winner_hash)
    if winner_session.test_evaluation_count != 0 or winner_session.result is not None or "test" in winner_session.metrics:
        raise PlotCacheError("搜索 winner 缓存意外包含最终测试状态。")

    batch_hashes = {
        "batch_snapshot_sha256": snapshot_sha256,
        "receipt_manifest_sha256": receipt_manifest_sha256,
        "receipt_file_sha256": receipt_file_sha256,
        "sequence_plan_sha256": getattr(result.sequence_plan, "plan_sha256", None),
    }
    sources: list[Any] = []
    unavailable: list[tuple[str, str]] = []
    for partition in ("train", "validation"):
        source = _session_partition_source(
            winner_session,
            partition,
            source_kind="batch_cache",
            owner_kind="batch_job",
            owner_id=job_id,
            source_id=job_id,
            manifest=result.snapshot.to_dict(),
            plot_manifest_version=winner_plot_manifest_version,
            batch_hashes=batch_hashes,
            snapshot=result.snapshot,
        )
        if source is not None:
            sources.append(source)
        else:
            unavailable.append((partition, f"winner 缓存没有可用的 {partition} 结果行。"))

    try:
        final_session, final_plot_manifest_version = _load_verified_batch_test_session(
            store=store,
            job=job,
            artifact_dir=artifact_dir,
            expected_snapshot=result.snapshot,
            expected_winner_trial_id=result.winner.trial_id,
        )
    except PlotCacheError as exc:
        unavailable.append(("test", str(exc)))
    else:
        source = _session_partition_source(
            final_session,
            "test",
            source_kind="batch_cache",
            owner_kind="batch_job",
            owner_id=job_id,
            source_id=job_id,
            manifest=result.snapshot.to_dict(),
            plot_manifest_version=final_plot_manifest_version,
            batch_hashes=batch_hashes,
            snapshot=result.snapshot,
        )
        if source is not None:
            sources.append(source)
        else:
            unavailable.append(("test", "已完成的最终测试 session 没有可用的测试结果行。"))
    return PlotSourceCollection(tuple(sources), tuple(unavailable))


def _load_verified_batch_test_session(
    *, store, job, artifact_dir: Path, expected_snapshot, expected_winner_trial_id: str,
):
    from .selection import FrozenSelection

    job_id = str(job["job_id"])
    permission = store.get_test_permission(job_id)
    if (
        not isinstance(permission, dict)
        or permission.get("state") != "completed"
        or not isinstance(permission.get("final_session_id"), str)
    ):
        raise PlotCacheError("test 分区不可用：history 没有完成的最终测试 permission。")
    result_payload = _json_object(permission.get("result_json"), "最终测试回执")
    final_session_id = permission["final_session_id"]
    if (
        result_payload.get("final_run_id") != final_session_id
        or result_payload.get("session_id") != final_session_id
        or result_payload.get("state") != "tested"
        or not _has_exact_test_evaluation_count(
            result_payload.get("test_evaluation_count"), 1
        )
    ):
        raise PlotCacheError("test 分区不可用：最终测试回执的 session 身份、状态或计数无效。")

    selection_path = (artifact_dir / "frozen-selection.joblib").resolve()
    final_path = (artifact_dir / "final-session.joblib").resolve()
    _assert_owned_path(selection_path, artifact_dir, "frozen selection")
    _assert_owned_path(final_path, artifact_dir, "final session")
    if not selection_path.is_file() or not final_path.is_file():
        raise PlotCacheError("test 分区不可用：owned frozen selection 或 final session 文件缺失。")
    try:
        selection = joblib.load(selection_path)
        envelope = joblib.load(final_path)
    except Exception as exc:
        raise PlotCacheError(f"test 分区不可用：无法读取最终测试缓存：{exc}") from exc
    if not isinstance(selection, FrozenSelection):
        raise PlotCacheError("test 分区不可用：owned frozen selection 类型不正确。")
    try:
        selection.verify()
    except Exception as exc:
        raise PlotCacheError(f"test 分区不可用：frozen selection 核验失败：{exc}") from exc
    if (
        selection.job_id != job_id
        or selection.winner_trial_id != expected_winner_trial_id
        or selection.snapshot.to_dict() != expected_snapshot.to_dict()
    ):
        raise PlotCacheError("test 分区不可用：frozen selection 不属于所选 job/winner/snapshot。")
    if (
        not isinstance(envelope, dict)
        or envelope.get("kind") != "pyml-workbench-internal-session"
        or envelope.get("schema_version") != 1
        or envelope.get("session_path") != str(final_path)
        or envelope.get("session_id") != final_session_id
        or envelope.get("state") != "tested"
        or envelope.get("selection_sha256") != selection.selection_sha256
        or envelope.get("selection") != selection.to_dict()
    ):
        raise PlotCacheError("test 分区不可用：final session 身份、路径或 selection 绑定无效。")
    session = envelope.get("session")
    if envelope.get("frozen_config_sha256") != getattr(session, "frozen_config_sha256", None):
        raise PlotCacheError("test 分区不可用：final session 冻结配置哈希不匹配。")
    plot_manifest_version = _verify_session_plot_cache_manifest(envelope, session)
    if not _test_cache_is_successful(session, envelope_state=envelope.get("state")):
        raise PlotCacheError("test 分区不可用：缓存没有成功的一次性最终测试结果。")
    if result_payload.get("frozen_config_sha256") != session.frozen_config_sha256:
        raise PlotCacheError("test 分区不可用：permission 记录的冻结配置哈希不匹配。")
    _validate_session(
        session,
        snapshot=expected_snapshot,
        expected_config_sha256=session.frozen_config_sha256,
    )
    return session, plot_manifest_version


def _session_partition_source(
    session,
    partition: str,
    *,
    source_kind: str,
    owner_kind: str,
    owner_id: str,
    source_id: str,
    manifest: Mapping[str, Any],
    plot_manifest_version: int,
    batch_hashes: Mapping[str, Any] | None = None,
    snapshot=None,
):
    from .plotting import (
        PlotColumn, PlotExplanationCache, PlotProvenance, PlotScoreCache,
        PlotSequenceMap, PlotSource,
    )

    if partition not in {"train", "validation", "test"}:
        raise PlotCacheError(f"Unsupported result partition {partition!r}")
    result = getattr(session, "result", None)
    result_frame = result.results if result is not None else pd.DataFrame(getattr(session, "rows", []))
    if not isinstance(result_frame, pd.DataFrame) or result_frame.empty or "split" not in result_frame or "row_position" not in result_frame:
        return None
    selected = result_frame.loc[result_frame["split"] == partition]
    if selected.empty:
        return None
    positions = tuple(_as_position(value) for value in selected["row_position"].tolist())
    if len(set(positions)) != len(positions):
        raise PlotCacheError(f"{partition} 缓存含重复 row_position，拒绝绘图。")
    features = getattr(session, "features", None)
    if not isinstance(features, pd.DataFrame) or any(position >= len(features) for position in positions):
        raise PlotCacheError(f"{partition} 缓存的 row_position 超出当前 session 数据范围。")
    split_positions = getattr(session, "splits", {}).get(partition)
    allowed_positions = set(int(value) for value in np.asarray(split_positions, dtype=np.int64).tolist())
    if not set(positions).issubset(allowed_positions):
        raise PlotCacheError(f"{partition} 结果行不属于该 session 的同名分区。")

    columns: list[Any] = []
    for index, label in enumerate(features.columns):
        columns.append(_plot_column(
            f"data.{index}", label, str(features.dtypes.iloc[index]),
            [features.iloc[position, index] for position in positions],
        ))
    target = getattr(session, "target", None)
    target_name = getattr(getattr(session, "config", None), "dataset", None)
    target_name = getattr(target_name, "target_column", None)
    if target is not None:
        columns.append(_plot_column(
            f"data.{len(columns)}", target_name if target_name is not None else "target",
            str(target.dtype), [target.iloc[position] for position in positions],
        ))

    outputs = _result_outputs(session.config.task, selected)
    score_cache = None
    score_unavailable_reason = None
    if plot_manifest_version == 1:
        if session.config.task == "classification":
            score_unavailable_reason = (
                "旧版 session manifest 未保护真实概率或决策分数，需重新训练后才能绘 ROC/PR；"
                "其他已有结果图仍可使用。"
            )
    else:
        raw_scores = getattr(session, "plot_scores", {})
        score_record = raw_scores.get(partition) if isinstance(raw_scores, Mapping) else None
        if isinstance(score_record, Mapping) and score_record.get("available") is True:
            score_positions = tuple(_as_position(value) for value in score_record.get("row_positions", ()))
            score_values = score_record.get("values")
            score_map = score_record.get("score_column_map")
            class_order = score_record.get("class_order")
            if (
                score_positions != positions
                or not isinstance(score_values, list)
                or not isinstance(score_map, list)
                or not isinstance(class_order, list)
            ):
                raise PlotCacheError(f"{partition} 分类分数的行位置、类别或列映射与结果行不匹配。")
            try:
                score_cache = PlotScoreCache(
                    row_positions=score_positions,
                    score_type=score_record["score_type"],
                    semantics=score_record["semantics"],
                    class_order=tuple(_scalar(value, context="评分类别", allow_tuple=True) for value in class_order),
                    score_column_map=tuple(
                        (
                            _as_position(item["column_index"]),
                            _scalar(item["class_label"], context="评分列类别", allow_tuple=True),
                        )
                        for item in score_map
                    ),
                    values=tuple(tuple(float(value) for value in row) for row in score_values),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise PlotCacheError(f"{partition} 分类分数缓存格式无效：{exc}") from exc
        elif isinstance(score_record, Mapping):
            score_unavailable_reason = str(score_record.get("reason") or "估计器没有可用的真实类别分数输出。")
        elif session.config.task == "classification":
            score_unavailable_reason = (
                "旧缓存没有真实概率或决策分数，需重新训练后才能绘 ROC/PR；其他已有结果图仍可使用。"
                if not hasattr(session, "plot_scores")
                else f"此 {partition} 分区没有真实分类分数缓存；请重新训练以生成 ROC/PR。"
            )
    if score_cache is not None and session.config.task == "classification":
        target_output = next(
            (column for column in outputs if column.column_id == "classification.y_true"),
            None,
        )
        if target_output is not None and len({
            (type(value), value) for value in target_output.values if not _is_missing(value)
        }) < 2:
            score_unavailable_reason = "当前评估分区只有一个类别；ROC/PR 需要正例和负例同时存在。"
    explanation_cache = None
    explanation_unavailable_reason = None
    if plot_manifest_version == 1:
        explanation_unavailable_reason = (
            "旧版 session manifest 未保护模型解释量，需重新训练后才能查看模型系数或特征重要性。"
        )
    else:
        raw_explanation = getattr(session, "plot_explanations", None)
        if isinstance(raw_explanation, Mapping) and raw_explanation.get("available") is True:
            try:
                explanation_cache = PlotExplanationCache(
                    kind=raw_explanation["kind"],
                    feature_names=tuple(str(value) for value in raw_explanation["feature_names"]),
                    class_labels=tuple(
                        _scalar(value, context="模型解释类别", allow_tuple=True)
                        for value in raw_explanation.get("class_labels", ())
                    ),
                    values=tuple(tuple(float(value) for value in row) for row in raw_explanation["values"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise PlotCacheError(f"模型解释缓存格式无效：{exc}") from exc
        elif isinstance(raw_explanation, Mapping):
            explanation_unavailable_reason = str(
                raw_explanation.get("reason") or "此估计器没有可用的原生特征解释量。"
            )
        else:
            explanation_unavailable_reason = "此旧缓存没有模型原生解释量；需重新训练后可查看。"
    plan = getattr(session, "sequence_plan", None)
    sequence_map: tuple[Any, ...] = ()
    if plan is not None:
        maps = []
        order_values = getattr(plan, "_order_values_ns", None)
        for _, row in selected.iterrows():
            position = _as_position(row["row_position"])
            source_positions = row.get("sequence_source_row_positions")
            if source_positions is None or _is_missing(source_positions):
                source_positions = (position,)
            if isinstance(source_positions, np.ndarray):
                source_positions = source_positions.tolist()
            if not isinstance(source_positions, (list, tuple)):
                raise PlotCacheError("序列缓存的窗口来源 row_positions 格式无效。")
            source_positions = tuple(_as_position(value) for value in source_positions)
            if any(value not in allowed_positions for value in source_positions):
                raise PlotCacheError("序列窗口引用了所选分区之外的源行。")
            target_position = _as_position(row.get("sequence_target_row_position", position))
            if target_position != position:
                raise PlotCacheError("序列缓存目标位置与结果 row_position 不一致。")
            group_value = None
            if position < len(plan._group_values_by_row):
                group_value = _scalar(plan._group_values_by_row[position], context="序列组值", allow_tuple=True)
            order_value_ns = None
            if order_values is not None:
                if position >= len(order_values):
                    raise PlotCacheError("序列时间顺序映射超出缓存行范围。")
                order_value_ns = int(order_values[position])
            maps.append(PlotSequenceMap(
                target_row_position=target_position,
                source_row_positions=source_positions,
                group_value=group_value,
                time_value=None,
                order_value_ns=order_value_ns,
            ))
        sequence_map = tuple(maps)

    batch_hashes = batch_hashes or {}
    provenance = PlotProvenance(
        source_sha256=manifest.get("source_sha256"),
        data_sha256=manifest.get("data_sha256"),
        split_sha256=manifest.get("split_sha256"),
        frozen_config_sha256=session.frozen_config_sha256,
        batch_snapshot_sha256=batch_hashes.get("batch_snapshot_sha256"),
        receipt_manifest_sha256=batch_hashes.get("receipt_manifest_sha256"),
        receipt_file_sha256=batch_hashes.get("receipt_file_sha256"),
        sequence_plan_sha256=batch_hashes.get("sequence_plan_sha256") or getattr(plan, "plan_sha256", None),
        cache_payload_sha256=_source_payload_sha256(
            positions, tuple(columns), outputs, sequence_map, score_cache, explanation_cache
        ),
    )
    return PlotSource(
        schema_version=1,
        source_kind=source_kind,
        source_id=source_id,
        owner_kind=owner_kind,
        owner_id=owner_id,
        partition=partition,
        row_positions=positions,
        columns=tuple(columns),
        outputs=outputs,
        sequence_map=sequence_map,
        provenance=provenance,
        partition_count=len(positions),
        score_cache=score_cache,
        explanation_cache=explanation_cache,
        score_unavailable_reason=score_unavailable_reason,
        explanation_unavailable_reason=explanation_unavailable_reason,
    )


def _result_outputs(task: str, selected: pd.DataFrame) -> tuple[Any, ...]:
    from .plotting import PlotColumn

    output_map: list[tuple[str, str]] = []
    if task in {"classification", "regression"}:
        output_map.extend(((f"{task}.y_true", "真实值"), (f"{task}.y_pred", "预测值")))
    elif task == "clustering":
        output_map.append(("clustering.label", "簇"))
    elif task == "dimensionality reduction":
        output_map.extend(
            (f"dimensionality_reduction.{name}", name.replace("_", " "))
            for name in selected.columns
            if name.startswith("component_")
        )
    elif task == "sequence_modeling":
        output_map.append(("hmm.hidden_state", "隐藏状态"))
        output_map.extend(
            (f"hmm.posterior.{int(name.removeprefix('posterior_')) - 1}", f"后验概率 {name.removeprefix('posterior_')}")
            for name in selected.columns
            if name.startswith("posterior_") and name.removeprefix("posterior_").isdigit()
        )
    elif task == "anomaly detection":
        output_map.append(("anomaly_detection.label", "异常标签"))

    names = {
        "classification.y_true": "y_true",
        "classification.y_pred": "y_pred",
        "regression.y_true": "y_true",
        "regression.y_pred": "y_pred",
        "clustering.label": "y_pred",
        "hmm.hidden_state": "hidden_state",
        "anomaly_detection.label": "y_pred",
    }
    outputs = []
    for column_id, label in output_map:
        name = names.get(column_id, column_id.rsplit(".", 1)[-1])
        if name not in selected.columns:
            continue
        series = selected[name]
        outputs.append(_plot_column(column_id, label, str(series.dtype), series.tolist()))
    return tuple(outputs)


def _validate_session(session, *, snapshot=None, expected_config_sha256: str):
    from .experiment import ExperimentSession, _canonical_config, _json_digest, _table_digest

    if not isinstance(session, ExperimentSession):
        raise PlotCacheError("session 缓存类型无效。")
    try:
        frozen_config, config_sha256 = _canonical_config(session.config)
    except Exception as exc:
        raise PlotCacheError(f"无法核验 session 冻结配置：{exc}") from exc
    if (
        config_sha256 != expected_config_sha256
        or config_sha256 != session.frozen_config_sha256
        or frozen_config != session.frozen_config
        or session.fitted_model.config != frozen_config
        or session.fitted_model.frozen_config_sha256 != config_sha256
    ):
        raise PlotCacheError("session 冻结配置、拟合模型与来源配置哈希不一致。")
    manifest = session.snapshot_manifest
    if not isinstance(manifest, dict):
        raise PlotCacheError("session 数据快照格式无效。")
    if manifest and session.source_path != manifest.get("source_path"):
        raise PlotCacheError("session 数据快照来源路径无效。")
    if snapshot is not None:
        try:
            snapshot.verify(session.config)
        except Exception as exc:
            raise PlotCacheError(f"owned 数据快照核验失败：{exc}") from exc
        if snapshot.to_dict() != manifest:
            raise PlotCacheError("session 数据快照与 owned snapshot manifest 不一致。")
        if _table_digest(session.features) != _table_digest(snapshot.features):
            raise PlotCacheError("session 特征与 owned snapshot 不一致。")
        if session.fit_scope == "train":
            if _table_digest(session.target) != _table_digest(snapshot.target):
                raise PlotCacheError("session 目标与 owned snapshot 不一致。")
            expected_splits = snapshot.splits
        elif session.fit_scope == "train_validation":
            if _table_digest(session.target) != _table_digest(snapshot.target):
                raise PlotCacheError("final session 目标与 owned snapshot 不一致。")
            expected_splits = {
                "train": np.concatenate([snapshot.splits["train"], snapshot.splits["validation"]]),
                "validation": np.array([], dtype=np.int64),
                "test": snapshot.splits["test"],
            }
        else:
            raise PlotCacheError("session fit_scope 不属于预期的批量缓存。")
        for partition, expected in expected_splits.items():
            observed = np.asarray(session.splits[partition], dtype=np.int64)
            if not np.array_equal(observed, np.asarray(expected, dtype=np.int64)):
                raise PlotCacheError(f"session {partition} 行位置与 owned snapshot 不一致。")
    else:
        if manifest and manifest.get("objective_labels_column") is not None:
            raise PlotCacheError("单任务 session 缺少可复核的 objective-label 快照，拒绝绘图。")
        observed_data_sha256 = _json_digest({
            "features": _table_digest(session.features),
            "target": _table_digest(session.target),
            "objective_labels": None,
        })
        if manifest.get("data_sha256") is not None and observed_data_sha256 != manifest.get("data_sha256"):
            raise PlotCacheError("session 数据哈希与缓存特征/目标不匹配。")
        positions = {
            name: [int(value) for value in np.asarray(session.splits[name], dtype=np.int64)]
            for name in ("train", "validation", "test")
        }
        observed_split_sha256 = _json_digest(positions)
        if manifest.get("split_sha256") is not None and observed_split_sha256 != manifest.get("split_sha256"):
            raise PlotCacheError("session split 哈希与缓存行位置不匹配。")
        expected_source = Path(session.config.dataset.source_path).expanduser().resolve()
        if Path(session.source_path).expanduser().resolve() != expected_source:
            raise PlotCacheError("session source path 与其冻结配置不一致。")
        manifest = dict(
            manifest,
            source_path=str(Path(session.source_path).expanduser().resolve()),
            data_sha256=observed_data_sha256,
            split_sha256=observed_split_sha256,
        )
    if manifest.get("source_sha256") is not None and not _is_sha256(str(manifest.get("source_sha256"))):
        raise PlotCacheError("session source_sha256 无效。")
    return manifest


def _has_exact_test_evaluation_count(value: Any, expected: int) -> bool:
    """Require a serialized test count to be an actual integer, not bool/float."""
    return type(value) is int and value == expected


def _test_cache_is_successful(session, *, envelope_state: str | None) -> bool:
    if session is None or envelope_state != "tested":
        return False
    task = getattr(getattr(session, "config", None), "task", None)
    required_capability = {
        "classification": "predict",
        "regression": "predict",
        "clustering": "predict",
        "dimensionality reduction": "transform",
        "anomaly detection": "predict",
        "sequence_modeling": "predict",
    }.get(task)
    capabilities = getattr(getattr(session, "fitted_model", None), "capabilities", None)
    if (
        required_capability is None
        or not isinstance(capabilities, Mapping)
        or capabilities.get(required_capability) is not True
    ):
        return False
    result = getattr(session, "result", None)
    audit = getattr(result, "audit", None)
    return bool(
        _has_exact_test_evaluation_count(session.test_evaluation_count, 1)
        and session.finalized is True
        and result is not None
        and isinstance(getattr(session, "metrics", None), dict)
        and "test" in session.metrics
        and isinstance(audit, dict)
        and _has_exact_test_evaluation_count(audit.get("test_evaluation_count"), 1)
        and "test" in getattr(result, "results", pd.DataFrame()).get("split", pd.Series(dtype=object)).tolist()
    )


def _plot_column(column_id: str, label: Any, dtype: str, values: list[Any]):
    from .plotting import PlotColumn

    safe_label = _scalar(label, context="列标签", allow_tuple=True)
    if not isinstance(safe_label, Hashable):
        raise PlotCacheError("列标签不可哈希，拒绝构造绘图来源。")
    safe_values = tuple(_scalar(value, context=f"列 {safe_label!r}") for value in values)
    return PlotColumn(column_id=str(column_id), label=safe_label, dtype=str(dtype), values=safe_values)


def _scalar(value: Any, *, context: str, allow_tuple: bool = False) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, tuple) and allow_tuple:
        return tuple(_scalar(item, context=context, allow_tuple=True) for item in value)
    if isinstance(value, (list, dict, set, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        raise PlotCacheError(f"{context} 含有嵌套或可变容器值。")
    if not isinstance(value, Hashable):
        raise PlotCacheError(f"{context} 的值不可哈希。")
    if not isinstance(value, (str, bytes, bool, int, float, Decimal, Fraction, UUID, date, datetime, time, timedelta, type(None))):
        raise PlotCacheError(f"{context} 含不支持的标量类型 {type(value).__name__!r}。")
    return value


def _as_position(value: Any) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlotCacheError("row_position 必须是非负整数。")
    return int(value)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _source_payload_sha256(
    row_positions, columns, outputs, sequence_map, score_cache=None, explanation_cache=None
) -> str:
    payload = {
        "row_positions": [int(value) for value in row_positions],
        "columns": [_column_payload(column) for column in columns],
        "outputs": [_column_payload(column) for column in outputs],
        "sequence_map": [
            {
                "target_row_position": int(item.target_row_position),
                "source_row_positions": [int(value) for value in item.source_row_positions],
                "group_value": _typed_value(item.group_value),
                "time_value": _typed_value(item.time_value),
                "order_value_ns": item.order_value_ns,
            }
            for item in sequence_map
        ],
        "score_cache": None if score_cache is None else {
            "row_positions": list(score_cache.row_positions),
            "score_type": score_cache.score_type,
            "semantics": score_cache.semantics,
            "class_order": [_typed_value(value) for value in score_cache.class_order],
            "score_column_map": [
                {"column_index": index, "class_label": _typed_value(label)}
                for index, label in score_cache.score_column_map
            ],
            "values": [list(row) for row in score_cache.values],
        },
        "explanation_cache": None if explanation_cache is None else {
            "kind": explanation_cache.kind,
            "feature_names": list(explanation_cache.feature_names),
            "class_labels": [_typed_value(value) for value in explanation_cache.class_labels],
            "values": [list(row) for row in explanation_cache.values],
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _column_payload(column):
    return {
        "column_id": column.column_id,
        "label": _typed_value(column.label),
        "dtype": column.dtype,
        "values": [_typed_value(value) for value in column.values],
    }


def _typed_value(value):
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, float):
        if math.isnan(value):
            encoded: Any = "nan"
        elif math.isinf(value):
            encoded = "inf" if value > 0 else "-inf"
        else:
            encoded = value
    elif isinstance(value, bytes):
        encoded = value.hex()
    elif isinstance(value, (datetime, date, time, timedelta)):
        encoded = value.isoformat() if not isinstance(value, timedelta) else value.total_seconds()
    elif isinstance(value, Decimal):
        encoded = str(value)
    elif isinstance(value, Fraction):
        encoded = [value.numerator, value.denominator]
    elif isinstance(value, UUID):
        encoded = str(value)
    elif isinstance(value, tuple):
        encoded = [_typed_value(item) for item in value]
    elif isinstance(value, (str, bool, int)):
        encoded = value
    else:
        raise PlotCacheError(f"无法稳定计算绘图缓存哈希：不支持的标量 {type(value).__name__!r}。")
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": encoded}


def _frame_sha256(frame: pd.DataFrame) -> str:
    try:
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    except TypeError as exc:
        raise PlotCacheError(f"无法为完整表格建立稳定数据哈希：{exc}") from exc
    columns = [{"label": _typed_value(_scalar(name, context="列标签", allow_tuple=True)), "dtype": str(dtype)}
               for name, dtype in zip(frame.columns, frame.dtypes)]
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(values)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PlotCacheError(f"{label} JSON 无效：{exc}") from exc
    if not isinstance(value, Mapping):
        raise PlotCacheError(f"{label} 必须是 JSON 对象。")
    return dict(value)


def _assert_owned_path(path: Path, owner_dir: Path, label: str) -> None:
    try:
        path.relative_to(owner_dir.resolve())
    except ValueError as exc:
        raise PlotCacheError(f"{label} 路径越出所选 job 的所有者目录。") from exc


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
