"""One-stop PYEDA conversion and explicit PYML machine-learning run.

The program is configuration driven. It requires an explicit target, task,
model and grouping column. Conversion and model artifacts are written to a
new run directory, and the final test is evaluated only after the fitted
configuration has been frozen.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import traceback
import tomllib
import uuid
from typing import Any, Callable


CORE_DIR = Path(__file__).resolve().parent
WINDOW_REGRESSION_MODELS = frozenset({"N04", "N06", "N07", "N08", "N09"})
MODULE_TO_PIP_PACKAGE = {
    "torch": "torch",
    "skorch": "skorch",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "statsmodels": "statsmodels",
    "hmmlearn": "hmmlearn",
    "umap": "umap-learn",
}


class PipelineError(RuntimeError):
    """Raised when the requested one-stop run cannot continue."""


class PipelineCancelled(PipelineError):
    """Raised when the operator cancels a run."""


ProgressCallback = Callable[[str, int, int, str], None]


def summarize_slot_status(counts: dict[str, int], total: int) -> str:
    completed = int(counts.get("completed", 0))
    unavailable = int(counts.get("unavailable", 0))
    failed = int(counts.get("failed", 0))
    if completed == total and total > 0:
        return "completed"
    if completed == 0:
        return "failed"
    if failed and unavailable:
        return "completed_with_failures_and_unavailable"
    if failed:
        return "completed_with_failures"
    if unavailable:
        return "completed_with_unavailable"
    return "completed_with_partial_results"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取 JSON：{path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"JSON 顶层必须是对象：{path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text + "\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pyml_root() -> Path | None:
    for parent in (CORE_DIR, *CORE_DIR.parents):
        if (parent / "src" / "pyml_workbench").is_dir():
            return parent
    return None


def ensure_pyml_importable(config: dict[str, Any]) -> None:
    root_value = config.get("pyml_root") or find_pyml_root()
    if root_value:
        root = Path(root_value).expanduser().resolve()
        src = root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        import pyml_workbench  # noqa: F401
    except ImportError as exc:
        raise PipelineError(
            "无法导入 pyml_workbench。请在配置中填写 pyml_root，或在当前 Python 环境安装 PYML。"
        ) from exc


def emit(callback: ProgressCallback | None, stage: str, current: int, total: int, detail: str) -> None:
    if callback is not None:
        callback(stage, current, total, detail)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("pyeda_plugin", "input", "output_root", "target_column", "task", "model_id", "group_column")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise PipelineError("配置缺少必需字段：" + ", ".join(missing))
    source = Path(config["input"]).expanduser().resolve()
    plugin = Path(config["pyeda_plugin"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    if not source.is_dir():
        raise PipelineError(f"PYEDA 输入目录不存在：{source}")
    if not plugin.is_dir():
        raise PipelineError(f"PYEDA 插件目录不存在：{plugin}")
    if source == output_root or source in output_root.parents:
        raise PipelineError("输出根目录不能位于 PYEDA 输入目录内")
    for rel in ("manifest.json", "summary.json", "intake/analysis_settings.json", "intake/feature_definitions.csv"):
        if not (source / rel).is_file():
            raise PipelineError(f"PYEDA 结果包缺少：{rel}")
    manifest = read_json(plugin / "pyml-plugin.json")
    if manifest.get("id") != "data-conversion" or manifest.get("source_project") != "PYEDA":
        raise PipelineError("所选目录不是 PYEDA 数据转换插件")
    entry = manifest.get("entrypoint")
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise PipelineError("PYEDA 插件缺少有效入口")
    entry_path = (plugin / entry["path"]).resolve()
    if not entry_path.is_file() or plugin not in entry_path.parents:
        raise PipelineError("PYEDA 插件入口缺失或位于插件目录外")
    if config["task"] not in {"regression", "classification"}:
        raise PipelineError("一站式机器学习入口当前要求 task 为 regression 或 classification")
    if not isinstance(config["model_id"], str) or not re.fullmatch(r"[A-Z][0-9]{2}", config["model_id"]):
        raise PipelineError("model_id 必须为 PYML 目录中的模型编号")
    conversion = dict(config.get("conversion") or {})
    conversion.setdefault("source", "participant_task_role")
    conversion.setdefault("include_outcomes", True)
    conversion.setdefault("only_valid_records", True)
    conversion.setdefault("verify_manifest_hashes", True)
    if conversion["source"] not in {"participant_task_role", "participant_exposure", "records"}:
        raise PipelineError("conversion.source 不受支持")
    if not str(config["target_column"]).startswith("target__"):
        raise PipelineError("target_column 必须是转换器明确导出的 target__ 列")
    settings = read_json(source / "intake" / "analysis_settings.json")
    outcome_names = settings.get("outcomes")
    if not isinstance(outcome_names, list) or not all(isinstance(name, str) for name in outcome_names):
        raise PipelineError("analysis_settings.json 的 outcomes 必须是字符串列表")
    known = {f"target__{name}" for name in outcome_names} | {"target__ioc_binary"}
    ml = dict(config.get("machine_learning") or {})
    ml.setdefault("parameters", {})
    ml.setdefault("seed", 42)
    ml.setdefault("feature_columns", None)
    configured_outcomes = ml.get("outcomes")
    if configured_outcomes is None:
        configured_outcomes = [config["target_column"]]
    if not isinstance(configured_outcomes, list) or not configured_outcomes:
        raise PipelineError("machine_learning.outcomes 必须是非空结果列列表")
    normalized_outcomes = []
    for value in configured_outcomes:
        if not isinstance(value, str):
            raise PipelineError("outcome 必须是字符串")
        column = value if value.startswith("target__") else f"target__{value}"
        if column not in known or not re.fullmatch(r"target__[\w.-]+", column) or column.endswith("."):
            raise PipelineError(f"未知或无效的结果列：{column}")
        if column not in normalized_outcomes:
            normalized_outcomes.append(column)
    if config["target_column"] not in normalized_outcomes:
        raise PipelineError("主 target_column 必须包含在 machine_learning.outcomes 中")
    if any(name != "target__ioc_binary" for name in normalized_outcomes) and not conversion["include_outcomes"]:
        raise PipelineError("表格建模需要 conversion.include_outcomes=true")
    ml["outcomes"] = normalized_outcomes
    outcome_tasks = dict(ml.get("outcome_tasks") or {})
    if "target__ioc_binary" in normalized_outcomes:
        outcome_tasks.setdefault("target__ioc_binary", "classification")
        if outcome_tasks["target__ioc_binary"] != "classification":
            raise PipelineError("IOC 事件标签必须使用分类任务")
    for outcome in normalized_outcomes:
        outcome_tasks.setdefault(outcome, config["task"])
    if any(task not in {"regression", "classification"} for task in outcome_tasks.values()):
        raise PipelineError("machine_learning.outcome_tasks 只能包含 regression 或 classification")
    if outcome_tasks[config["target_column"]] != config["task"]:
        raise PipelineError("主 task 必须与 target_column 对应的 outcome_tasks 一致")
    ml["outcome_tasks"] = outcome_tasks
    outcome_group_columns = dict(ml.get("outcome_group_columns") or {})
    if "target__ioc_binary" in normalized_outcomes:
        outcome_group_columns.setdefault("target__ioc_binary", "meta__group_id")
    ml["outcome_group_columns"] = outcome_group_columns
    ioc_classification = dict(ml.get("ioc_classification") or {})
    ioc_classification.setdefault("source", "intake/response_events.csv")
    ioc_classification.setdefault("target_column", "ioc")
    ioc_classification.setdefault("join_keys", ["group_id", "task", "participant_label"])
    ml["ioc_classification"] = ioc_classification
    sequence = ml.get("eda_sequence")
    if sequence is not None:
        if not isinstance(sequence, dict) or not all(
            type(sequence.get(key)) is int and sequence[key] > 0 for key in ("window", "horizon")
        ):
            raise PipelineError("eda_sequence 必须明确提供正整数 window 和 horizon")
    ml.setdefault("methods", [config["model_id"]])
    evaluation_mode = ml.setdefault("evaluation_mode", "validation")
    if evaluation_mode not in {"validation", "final"}:
        raise PipelineError("evaluation_mode 只支持 validation 或 final")
    ml.setdefault("iterations", 1 if evaluation_mode == "final" else 3)
    if type(ml["iterations"]) is not int or ml["iterations"] < 1:
        raise PipelineError("machine_learning.iterations 必须是正整数")
    if evaluation_mode == "final" and (
        ml["iterations"] != 1 or normalized_outcomes != [config["target_column"]]
        or ml["methods"] != [config["model_id"]] or ml.get("confirm_final_holdout") is not True
    ):
        raise PipelineError("最终测试需要冻结一个 outcome 和模型、执行一轮，并显式确认已独立保留的测试集")
    if not isinstance(config["group_column"], str) or not config["group_column"].startswith("meta__"):
        raise PipelineError("group_column 必须为转换表的 meta__ 身份字段")
    if any(not isinstance(value, str) or not value.startswith("meta__") for value in outcome_group_columns.values()):
        raise PipelineError("每个 outcome 的分组列必须为 meta__ 身份字段")
    if ml.get("time_column") is not None and (
        not isinstance(ml["time_column"], str) or not ml["time_column"].startswith("meta__")
    ):
        raise PipelineError("time_column 必须是原表中明确登记的 meta__ 时间字段")
    executable = config.get("python")
    if executable and not Path(executable).expanduser().resolve().is_file():
        raise PipelineError("指定的 Python 程序不存在")
    if executable and _console_python(Path(executable)) != _console_python(Path(sys.executable)):
        raise PipelineError("请使用配置指定的 Python 启动一站式程序，转换、训练及报告须共用该环境")
    ml.setdefault("all_missing_feature_policy", "constant_zero_with_indicator")
    if ml["all_missing_feature_policy"] not in {"constant_zero_with_indicator", "fail"}:
        raise PipelineError("all_missing_feature_policy 仅支持 constant_zero_with_indicator 或 fail")
    config = dict(config)
    config.update({"input": str(source), "pyeda_plugin": str(plugin), "output_root": str(output_root)})
    config["conversion"] = conversion
    config["machine_learning"] = ml
    return config


def _console_python(executable: Path) -> Path:
    executable = executable.expanduser().resolve()
    if executable.name.lower() == "pythonw.exe" and executable.with_name("python.exe").is_file():
        return executable.with_name("python.exe")
    return executable


def _stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def plugin_command(config: dict[str, Any], plugin_manifest: dict[str, Any]) -> list[str]:
    entry = plugin_manifest["entrypoint"]
    path = Path(config["pyeda_plugin"]) / entry["path"]
    if entry.get("kind") == "executable":
        return [str(path)]
    if entry.get("kind") == "python":
        python = str(_console_python(Path(sys.executable)))
        return [python, "-X", "utf8", str(path)]
    raise PipelineError(f"不支持的 PYEDA 插件入口类型：{entry.get('kind')}")


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / f"PYEDA-ML-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run.mkdir()
    return run


def save_state(run: Path, status: str, **extra: Any) -> None:
    payload = {"status": status, "updated_at": now(), **extra}
    write_json(run / "run-state.json", payload)


def run_conversion(
    config: dict[str, Any],
    run: Path,
    callback: ProgressCallback | None,
    cancel: threading.Event | None,
    *,
    directory: str = "conversion",
) -> dict[str, Any]:
    plugin_manifest = read_json(Path(config["pyeda_plugin"]) / "pyml-plugin.json")
    conversion = config["conversion"]
    conversion_dir = run / directory
    command = plugin_command(config, plugin_manifest)
    command.extend(["--input", config["input"], "--output", str(conversion_dir), "--source", conversion["source"], "--no-open"])
    command.append("--include-outcomes" if conversion["include_outcomes"] else "--no-include-outcomes")
    command.append("--only-valid-records" if conversion["only_valid_records"] else "--no-only-valid-records")
    if not conversion["verify_manifest_hashes"]:
        command.append("--no-verify-hashes")
    log_path = run / f"{directory}.log"
    save_state(run, "conversion_running", command=command, log=str(log_path))
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=str(Path(config["pyeda_plugin"])),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        while process.poll() is None:
            if cancel is not None and cancel.is_set():
                _stop_process(process)
                raise PipelineCancelled("用户取消了 PYEDA 转换")
            # The plugin atomically replaces progress.json. Windows readers can block the replace.
            try:
                with (conversion_dir / "progress.log").open("rb") as progress:
                    progress.seek(max(0, os.fstat(progress.fileno()).st_size - 4096))
                    lines = progress.read().decode("utf-8", errors="replace").splitlines()
                if lines:
                    _, stage, counts, detail = lines[-1].split("\t", 3)
                    current, total = map(int, counts.split("/"))
                    emit(callback, "PYEDA 转换", current, total, f"{stage} · {detail}")
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
    summary_path = conversion_dir / "batch_summary.json"
    deadline = time.monotonic() + 20
    while not summary_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not summary_path.is_file():
        raise PipelineError(f"PYEDA 转换没有生成 batch_summary.json，详见 {log_path}")
    summary = read_json(summary_path)
    if process.returncode != 0 or summary.get("status") != "completed_supported_tables":
        raise PipelineError(f"PYEDA 转换失败，状态为 {summary.get('status')}，详见 {log_path}")
    if conversion["verify_manifest_hashes"]:
        recorded = read_json(conversion_dir / "provenance.json").get("source_hashes", [])
        if not recorded or any(
            item.get("status") != "verified" for item in recorded if item.get("path") != "manifest.json"
        ):
            raise PipelineError("转换器未能证实所有选中来源文件与清单一致")
    emit(callback, "PYEDA 转换", 1, 1, str(conversion_dir))
    save_state(run, "converted", conversion_summary=summary)
    return summary


def _read_text_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if not header or any(not value for value in header) or len(header) != len(set(header)):
        raise PipelineError(f"CSV 表头缺失或重复：{path.name}")
    return header


def _read_text_csv(path: Path):
    import pandas as pd

    _read_text_csv_header(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, index_col=False)


def _read_hashed_intake(config: dict[str, Any], relative_path: str):
    root = Path(config["input"])
    relative = Path(relative_path)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise PipelineError(f"来源文件缺失或超出 PYEDA 输入包：{relative_path}")
    manifest = read_json(root / "manifest.json")
    key = path.relative_to(root).as_posix()
    expected = [item.get("sha256") for item in manifest.get("artifacts", [])
                if isinstance(item, dict) and str(item.get("path", "")).replace(chr(92), "/") == key]
    before = sha256(path)
    verify = config["conversion"]["verify_manifest_hashes"]
    if verify and (len(expected) != 1 or str(expected[0]).lower() != before):
        raise PipelineError(f"来源文件缺少唯一匹配的清单哈希：{relative_path}")
    frame = _read_text_csv(path)
    if sha256(path) != before:
        raise PipelineError(f"读取时来源发生变化：{relative_path}")
    return frame, {"path": key, "sha256": before, "verified": verify}


def _numeric_feature(values):
    import numpy as np
    import pandas as pd

    return pd.to_numeric(values, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def _converted_table(directory: Path):
    table = _read_text_csv(directory / "dataset.csv")
    mask = _read_text_csv(directory / "valid_mask.csv")
    glossary = read_json(directory / "feature_dictionary.json")
    entries = glossary.get("features")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise PipelineError("转换特征字典无效")
    listed = [entry.get("output_column") for entry in entries]
    if not listed or not all(isinstance(name, str) and name.startswith("feature__") for name in listed) or len(listed) != len(set(listed)):
        raise PipelineError("转换字典的已登记特征缺失、重复或命名无效")
    if ("sample_id" not in table or list(mask.columns) != ["sample_id", *[name for name in table.columns if name.startswith(("feature__", "target__"))]]
            or [name for name in table.columns if name.startswith("feature__")] != listed
            or not mask.drop(columns="sample_id").isin(["0", "1"]).all().all()
            or not table["sample_id"].str.strip().ne("").all()
            or table["sample_id"].duplicated().any()
            or not table["sample_id"].equals(mask["sample_id"])):
        raise PipelineError("数据表、特征字典与有效性掩码的表头、取值或样本顺序不一致")
    for name in listed:
        values = _numeric_feature(table[name])
        if not values.notna().eq(mask[name].eq("1")).all():
            raise PipelineError(f"特征有效性掩码与数据不一致：{name}")
        table[name] = values
    for name in table.columns:
        if name.startswith("target__") and not _numeric_feature(table[name]).notna().eq(mask[name].eq("1")).all():
            raise PipelineError(f"结局有限值掩码与数据不一致：{name}")
    return table, listed


def prepare_ml_dataset(config: dict[str, Any], run: Path, callback: ProgressCallback | None) -> dict[str, Any]:
    import pandas as pd

    conversion_dir = run / "conversion"
    dataset_path = conversion_dir / "dataset.csv"
    frame, listed = _converted_table(conversion_dir)
    ml = config["machine_learning"]
    target = config["target_column"]
    outcomes = ml["outcomes"]
    tasks = ml["outcome_tasks"]
    group = config["group_column"]
    if group not in frame.columns:
        raise PipelineError(f"分组列不存在：{group}")
    requested = ml.get("feature_columns")
    if requested is None:
        requested = listed
    if (not isinstance(requested, list) or not requested
            or any(not isinstance(name, str) or name not in listed for name in requested)
            or len(requested) != len(set(requested))):
        raise PipelineError("机器学习特征须为特征字典内互不重复的已登记列")

    numeric = frame.loc[:, requested].copy()
    all_missing_features = []
    for name in requested:
        if numeric[name].notna().any():
            continue
        if ml["all_missing_feature_policy"] == "fail":
            raise PipelineError(f"机器学习特征 {name} 全部缺失")
        indicator = f"{name}__missing_indicator"
        if indicator in frame:
            raise PipelineError(f"生成的缺失指示列与源字段重名：{indicator}")
        numeric[name] = 0.0
        numeric[indicator] = 1
        all_missing_features.append({"column": name, "indicator_column": indicator,
                                     "reason": "all_missing_in_converted_dataset",
                                     "policy": ml["all_missing_feature_policy"]})
    selected = list(numeric.columns)
    metadata = [name for name in frame.columns if name.startswith("meta__")]
    prepared = pd.concat([frame[["sample_id", *metadata]], numeric], axis=1)
    ml_dir = run / "machine-learning"
    outcome_paths, outcome_rows, outcome_group_counts, class_counts, excluded = {}, {}, {}, {}, {}
    tabular_valid = pd.Series(False, index=frame.index)
    event_audit = {}
    for outcome in outcomes:
        group_col = (ml.get("outcome_group_columns") or {}).get(outcome, group)
        if outcome == "target__ioc_binary":
            outcome_frame, event_audit = _prepare_ioc(config, frame, prepared, selected)
        else:
            if outcome not in frame:
                raise PipelineError(f"转换结果缺少机器学习 outcome：{outcome}")
            values = (_numeric_feature(frame[outcome]) if tasks[outcome] == "regression"
                      else frame[outcome].where(frame[outcome].str.strip().ne("")))
            prepared[outcome] = values
            if group_col not in prepared:
                raise PipelineError(f"结局 {outcome} 的分组列不存在：{group_col}")
            valid = values.notna() & prepared[group_col].str.strip().ne("")
            tabular_valid |= valid
            outcome_frame = prepared.loc[valid, ["sample_id", *metadata, *selected, outcome]].copy()
            if tasks[outcome] == "classification":
                # Prefix values so the downstream CSV reader preserves NA, leading zeros and other literal labels.
                outcome_frame[outcome] = "class__" + outcome_frame[outcome]
        if group_col not in outcome_frame or outcome_frame.empty:
            raise PipelineError(f"结局 {outcome} 缺少有效观测或分组列 {group_col}")
        if tasks[outcome] == "classification":
            if outcome_frame[outcome].nunique() < 2:
                raise PipelineError(f"结局 {outcome} 至少需要两个有效类别")
            class_counts[outcome] = {str(key): int(value) for key, value in outcome_frame[outcome].value_counts().items()}
        directory = ml_dir / "outcomes" / outcome
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ml_dataset.csv"
        outcome_frame.to_csv(path, index=False, encoding="utf-8-sig")
        outcome_paths[outcome] = str(path)
        outcome_rows[outcome] = len(outcome_frame)
        outcome_group_counts[outcome] = int(outcome_frame[group_col].nunique())
        excluded[outcome] = event_audit["excluded_events"] if outcome == "target__ioc_binary" else len(frame) - len(outcome_frame)

    if outcomes == ["target__ioc_binary"]:
        tabular_valid = prepared[group].str.strip().ne("")
    prepared = prepared.loc[tabular_valid]
    ml_dataset = ml_dir / "ml_dataset_all_outcomes.csv"
    prepared.to_csv(ml_dataset, index=False, encoding="utf-8-sig")
    prep = {
        "source_dataset": str(dataset_path), "output_dataset": str(ml_dataset),
        "input_rows": len(frame), "output_rows": len(prepared), "excluded_target_rows": len(frame) - len(prepared),
        "outcomes": outcomes, "outcome_tasks": tasks, "outcome_rows": outcome_rows, "outcome_paths": outcome_paths,
        "outcome_group_counts": outcome_group_counts, "outcome_excluded_rows": excluded,
        "classification_rows": {key: outcome_rows[key] for key in class_counts},
        "classification_class_counts": class_counts,
        "classification_label_map": {name: {key: key.removeprefix("class__") for key in counts}
                                     for name, counts in class_counts.items()},
        "ioc": event_audit,
        "requested_features": list(requested), "selected_features": selected, "all_missing_features": all_missing_features,
        "all_missing_feature_policy": ml["all_missing_feature_policy"], "excluded_features": [],
        "target_column": target, "group_column": group, "group_count": int(prepared[group].nunique()),
    }
    write_json(ml_dir / "data-preparation.json", prep)
    emit(callback, "准备机器学习数据", 1, 1, str(ml_dataset))
    return {"path": ml_dataset, "outcome_paths": outcome_paths, "features": selected, "prep": prep}


def _prepare_ioc(config, frame, prepared, selected):
    cfg = config["machine_learning"]["ioc_classification"]
    events, receipt = _read_hashed_intake(config, cfg["source"])
    keys = cfg["join_keys"]
    if not isinstance(keys, list) or not keys or len(keys) != len(set(keys)):
        raise PipelineError("IOC 连接需要互不重复的非空列名")
    pool_keys = [f"meta__{name}" for name in keys]
    label = cfg["target_column"]
    required = [*keys, "event_id", label]
    if any(name not in events for name in required) or any(name not in frame for name in pool_keys):
        raise PipelineError("IOC 事件表或特征表缺少配置的连接列、事件 ID 或标签")
    labels = _numeric_feature(events[label])
    present = events[label].str.strip().ne("")
    if (present & ~labels.isin([0, 1])).any():
        raise PipelineError("IOC 已提供的标签必须是 0 或 1")
    labelled = events.loc[labels.isin([0, 1]), required].copy()
    event_keys = [*keys, "event_id"]
    if labelled[event_keys].apply(lambda col: col.str.strip().eq("")).any().any() or labelled.duplicated(event_keys).any():
        raise PipelineError("IOC 已标注事件的身份键为空或重复")
    labelled["target__ioc_binary"] = labels.loc[labelled.index].astype(int)
    labelled = labelled.rename(columns=dict(zip(keys, pool_keys, strict=True)))
    metadata = [name for name in prepared if name.startswith("meta__")]
    pool = prepared.loc[prepared[pool_keys].apply(lambda col: col.str.strip().ne("")).all(axis=1)]
    grouped = pool.groupby(pool_keys, sort=False, as_index=False)
    identity_columns = [name for name in ("meta__subject_id", "meta__group_id", "meta__task", "meta__participant_label")
                        if name in metadata and name not in pool_keys]
    if identity_columns and grouped[identity_columns].nunique().loc[:, identity_columns].gt(1).any().any():
        raise PipelineError("IOC 连接键对应多个身份，无法安全聚合")
    pool = grouped.agg({**{name: "first" for name in identity_columns}, **dict.fromkeys(selected, "mean")})
    aligned = labelled.merge(pool, on=pool_keys, how="left", validate="many_to_one", indicator=True)
    misses = int(aligned["_merge"].eq("left_only").sum())
    matched = aligned.loc[aligned["_merge"].eq("both")].copy()
    matched["sample_id"] = [
        "ioc_event__" + hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
        for values in matched[[*pool_keys, "event_id"]].itertuples(index=False, name=None)
    ]
    keep = list(dict.fromkeys(["sample_id", *pool_keys, *identity_columns, *selected, "target__ioc_binary"]))
    audit = {"source": receipt, "events": len(events), "eligible_events": len(labelled),
             "matched_events": len(matched), "unmatched_labeled_events": misses,
             "match_rate": len(matched) / len(labelled) if len(labelled) else None,
             "excluded_events": len(events) - len(matched), "join_keys": keys,
             "feature_timing": "汇总特征按参与者任务匹配，评估关联，不作为时点预测证据"}
    return matched.loc[:, keep], audit


def _resolve_method_specs(
    config: dict[str, Any],
    *,
    task: str | None = None,
    methods_override: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve the configured model policy while retaining unavailable entries."""
    ensure_pyml_importable(config)
    from pyml_workbench import get_model, list_models

    requested_task = task or config["task"]
    methods = methods_override if methods_override is not None else config["machine_learning"].get("methods", "all_regression")
    if isinstance(methods, dict):
        methods = methods.get(requested_task) or methods.get("all") or ("all_regression" if requested_task == "regression" else "all_classification")
    if isinstance(methods, str) and methods in {"all", "all_task", "all_regression", "all_classification"}:
        catalog_task = "regression" if methods == "all_regression" else ("classification" if methods == "all_classification" else requested_task)
        specs = list_models(task=catalog_task, include_deferred=True)
        return sorted(specs, key=lambda item: str(item.get("id")))
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Z][0-9]{2}", name) for name in methods)
            or len(methods) != len(set(methods))):
        raise PipelineError("machine_learning.methods 必须为 all_regression、all_classification、all_task、all 或非空模型 ID 列表")
    specs = []
    for model_id in methods:
        try:
            spec = get_model(str(model_id))
        except Exception as exc:
            specs.append({"id": str(model_id), "task": requested_task, "implementation_status": "unavailable", "unavailable_reason": str(exc)})
            continue
        if spec.get("task") != requested_task:
            spec["implementation_status"] = "unavailable"
            spec["unavailable_reason"] = f"模型属于 {spec.get('task')}，不适用于 {requested_task}"
        specs.append(spec)
    return specs


def prepare_sequence_data(
    config: dict[str, Any], run: Path, ml_data: dict[str, Any], outcomes: list[str],
    callback: ProgressCallback | None, cancel: threading.Event | None,
) -> dict[str, dict[str, Any]]:
    """Join recorded responses to task-level targets without mixing subjects or tasks in a window."""
    import numpy as np
    import pandas as pd

    ml = config["machine_learning"]
    sequence = ml["eda_sequence"]
    window, horizon = sequence["window"], sequence["horizon"]
    record_config = {
        **config,
        "conversion": {**config["conversion"], "source": "records", "include_outcomes": False, "only_valid_records": True},
    }
    run_conversion(record_config, run, callback, cancel, directory="sequence-conversion")
    input_hashes = {}

    def read_intake(relative_path: str):
        frame, receipt = _read_hashed_intake(config, relative_path)
        input_hashes[relative_path] = receipt
        return frame

    emit(callback, "准备 EDA 回答序列", 0, len(outcomes), "核对逐记录 EDA、事件顺序和结局")
    records, _ = _converted_table(run / "sequence-conversion")
    records = records.loc[records["meta__window"].eq("turn_interval") & records["meta__role"].eq("self")].copy()
    events = read_intake("intake/response_events.csv")
    targets = read_intake("intake/individual_outcomes.csv")
    stream_keys = ["meta__group_id", "meta__task", "meta__subject_id", "meta__recording_id"]
    event_keys = ["meta__group_id", "meta__task", "meta__event_id"]
    rename_keys = {name: f"meta__{name}" for name in ("group_id", "task", "event_id", "participant_label")}
    events = events.rename(columns=rename_keys)
    speaker = events["meta__participant_label"]
    events = events.assign(event_speaker=speaker)[[*event_keys, "event_speaker", "timestamp_ms", "idea_sequence"]]
    records = records.merge(events, on=event_keys, how="left", validate="many_to_one")
    records["timestamp_ms"] = pd.to_numeric(records["timestamp_ms"], errors="coerce")
    order = pd.to_numeric(records["meta__sequence"], errors="coerce")
    if (records.empty or records[stream_keys].apply(lambda col: col.str.strip().eq("")).any().any()
            or not np.isfinite(records["timestamp_ms"]).all()
            or not np.isfinite(order).all()
            or not order.eq(pd.to_numeric(records["idea_sequence"], errors="coerce")).all()
            or not records["meta__participant_label"].eq(records["event_speaker"]).all()
            or records.duplicated([*stream_keys, "timestamp_ms"]).any()
            or records.duplicated([*stream_keys, "meta__event_id"]).any()):
        raise PipelineError("逐记录 EDA 的身份、本人角色或真实回答顺序无法唯一核验")
    if records.groupby("meta__subject_id")["meta__group_id"].nunique().gt(1).any():
        raise PipelineError("同一被试跨多个团队，当前序列方案无法同时保持两类分区隔离")
    records["meta__sequence_id"] = records.groupby(stream_keys, sort=True).ngroup()
    records = records.sort_values(["meta__sequence_id", "timestamp_ms"])
    join_keys = ["meta__group_id", "meta__task", "meta__participant_label"]
    targets = targets.rename(columns={**rename_keys, **{outcome[8:]: outcome for outcome in outcomes}})
    records = records.merge(targets[[*join_keys, *outcomes]], on=join_keys, how="left", validate="many_to_one")
    for name in ml_data["features"]:
        if name.endswith("__missing_indicator") and name not in records:
            values = pd.to_numeric(records[name.removesuffix("__missing_indicator")], errors="coerce")
            records[name] = (~np.isfinite(values)).astype(int)
        else:
            records[name] = pd.to_numeric(records[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Keep all input columns. A source-wide missing column has no observed value to impute from.
    for name in ml_data["features"]:
        if records[name].isna().all():
            if ml["all_missing_feature_policy"] == "fail":
                raise PipelineError(f"序列特征 {name} 全部缺失")
            records[name] = 0.0

    prepared = {}
    for index, outcome in enumerate(outcomes, 1):
        if cancel is not None and cancel.is_set():
            raise PipelineCancelled("用户取消了 EDA 回答序列准备")
        records[outcome] = pd.to_numeric(records[outcome], errors="coerce")
        eligible = records.loc[np.isfinite(records[outcome])].copy()
        lengths = eligible.groupby("meta__sequence_id")["sample_id"].transform("size")
        usable = eligible.loc[lengths.ge(window + horizon)].copy()
        directory = run / "machine-learning" / "sequence-outcomes" / outcome
        directory.mkdir(parents=True, exist_ok=True)
        columns = ["sample_id", *stream_keys, "meta__sequence_id", "meta__event_id", "timestamp_ms", *ml_data["features"], outcome]
        ordered_paths = {}
        for iteration in range(1, ml["iterations"] + 1):
            seed = int(ml["seed"]) + iteration - 1
            ordered = order_sequence_teams(usable, seed)
            path = directory / f"iteration-{iteration:02d}.csv"
            ordered.to_csv(path, columns=columns, index=False, encoding="utf-8-sig")
            ordered_paths[iteration] = str(path)
        prepared[outcome] = {
            "paths": ordered_paths, "rows": int(len(usable)),
            "sequences": int(usable["meta__sequence_id"].nunique()),
            "teams": int(usable["meta__group_id"].nunique()),
            "windows": int(len(usable) - (window + horizon - 1) * usable["meta__sequence_id"].nunique()),
            "excluded_missing_target_rows": int(len(records) - len(eligible)),
            "excluded_short_sequence_rows": int(len(eligible) - len(usable)),
        }
        emit(callback, "准备 EDA 回答序列", index, len(outcomes), outcome)
    ml_data["prep"]["sequence_regression"] = {
        "window": window, "horizon": horizon, "role": "self", "eda_window": "turn_interval",
        "input_hashes": input_hashes, "outcomes": prepared,
        "order": "team blocks ordered by iteration seed; strictly increasing posting time within each recording and task",
        "target": "participant-task creativity summary repeated across eligible response windows",
    }
    write_json(run / "machine-learning" / "data-preparation.json", ml_data["prep"])
    return prepared


def order_sequence_teams(frame: Any, seed: int) -> Any:
    """Find a team-block order whose native SequencePlan cut points fall on team boundaries."""
    import numpy as np

    team_column = "meta__group_id"
    sizes = frame.groupby([team_column, "meta__sequence_id"], sort=True).size()
    team_sizes = {team: block.tolist() for team, block in sizes.groupby(level=0, sort=True)}
    teams = list(team_sizes)
    if len(teams) < 3:
        raise PipelineError("序列回归需要至少三个有足够有效回答的独立团队")
    rng = random.Random(seed)
    for _ in range(2048):
        rng.shuffle(teams)
        lengths = [length for team in teams for length in team_sizes[team]]
        boundaries = set(np.cumsum([len(team_sizes[team]) for team in teams]))
        cumulative = np.cumsum(lengths)
        total = int(cumulative[-1])
        cut1 = min(range(1, len(lengths) - 1), key=lambda i: (abs(cumulative[i - 1] - 0.6 * total), i))
        cut2 = min(range(cut1 + 1, len(lengths)), key=lambda i: (abs(cumulative[i - 1] - 0.8 * total), i))
        if cut1 in boundaries and cut2 in boundaries:
            order = {team: index for index, team in enumerate(teams)}
            return frame.assign(team_order=frame[team_column].map(order)).sort_values(
                ["team_order", "meta__sequence_id", "timestamp_ms"]
            )
    raise PipelineError("当前序列长度无法生成通过原生划分核验的团队隔离分区")


def check_runtime_dependencies(config, run, callback, cancel):
    """Record the exact environment and pinned optional requirements without mutating it."""
    from importlib import metadata
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    ensure_pyml_importable(config)
    ml = config["machine_learning"]
    tasks = {ml["outcome_tasks"][outcome] for outcome in ml["outcomes"]}
    unavailable = {}
    for task in sorted(tasks):
        for spec in _resolve_method_specs(config, task=task):
            for module in spec.get("unavailable_modules") or []:
                unavailable.setdefault(module, []).append(str(spec["id"]))
    root = config.get("pyml_root") or find_pyml_root()
    project = Path(root) / "pyproject.toml" if root else None
    if project and project.is_file():
        data = tomllib.loads(project.read_text(encoding="utf-8"))["project"]
        requirements = [*data["dependencies"], *(item for values in data["optional-dependencies"].values() for item in values)]
    else:
        requirements = metadata.requires("pyml-workbench") or []
    pinned = {}
    for raw in requirements:
        requirement = Requirement(raw)
        if any(item.operator == "==" and not item.version.endswith(".*") for item in requirement.specifier):
            pinned[canonicalize_name(requirement.name)] = f"{requirement.name}{requirement.specifier}"

    python = str(_console_python(Path(sys.executable)))
    optional = sorted({pinned[canonicalize_name(MODULE_TO_PIP_PACKAGE[module])]
                       for module in unavailable if module in MODULE_TO_PIP_PACKAGE
                       and canonicalize_name(MODULE_TO_PIP_PACKAGE[module]) in pinned})
    plan = {"python": python, "requested_tasks": sorted(tasks), "unavailable_modules": unavailable,
            "pinned_optional_requirements": optional, "report_requirements": str(CORE_DIR / "requirements.txt")}
    write_json(run / "dependency-plan.json", plan)
    if cancel is not None and cancel.is_set():
        raise PipelineCancelled("用户取消了依赖检查")
    probe = subprocess.run(
        [python, "-X", "utf8", "-c", "import docx; import matplotlib; import matplotlib.backends.backend_agg"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    log = run / "dependency-check.log"
    log.write_text(probe.stdout + probe.stderr, encoding="utf-8")
    plan["status"] = ("report_dependencies_missing" if probe.returncode else
                      "optional_dependencies_unavailable" if unavailable else "available")
    write_json(run / "dependency-status.json", plan)
    if probe.returncode:
        raise PipelineError(f"报告依赖不可用。请在当前 PYML 环境安装 {CORE_DIR / 'requirements.txt'}，详见 {log}")
    emit(callback, "运行环境核对", 1, 1, "已核对报告依赖与可选模型")
    return plan


def run_machine_learning(
    config: dict[str, Any],
    run: Path,
    ml_data: dict[str, Any],
    callback: ProgressCallback | None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    ensure_pyml_importable(config)
    from pyml_workbench import DatasetConfig, ExperimentConfig, SplitConfig, evaluate_test, freeze_experiment, prepare_experiment
    import pandas as pd
    from pyml_workbench.experiment import build_extended_snapshot
    from pyml_workbench.sequence import SequenceConfig

    ml = config["machine_learning"]
    outcomes = list(ml_data["prep"]["outcomes"])
    outcome_tasks = dict(ml.get("outcome_tasks") or {})
    outcome_group_columns = dict(ml.get("outcome_group_columns") or {})
    method_specs_by_outcome: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        task = outcome_tasks.get(outcome, config["task"])
        method_specs_by_outcome[outcome] = _resolve_method_specs(config, task=task)
    sequence_outcomes = [
        outcome for outcome, specs in method_specs_by_outcome.items()
        if any(spec["id"] in WINDOW_REGRESSION_MODELS and spec.get("runtime_available", True) for spec in specs)
    ]
    sequence_data = {}
    sequence_error = None
    if sequence_outcomes and ml.get("eda_sequence") is not None:
        try:
            sequence_data = prepare_sequence_data(config, run, ml_data, sequence_outcomes, callback, cancel)
        except PipelineCancelled:
            raise
        except (OSError, ValueError, PipelineError, KeyError) as exc:
            sequence_error = str(exc)
            write_json(run / "machine-learning" / "sequence-error.json", {
                "error": sequence_error, "traceback": traceback.format_exc(),
            })

    def parameters_for(task: str, model_id: str) -> dict[str, Any]:
        task_parameters = (ml.get("parameters_by_task") or {}).get(task, ml.get("parameters") or {})
        return {**task_parameters, **(ml.get("parameters_by_model") or {}).get(model_id, {})}
    iterations = ml["iterations"]
    final_test = ml["evaluation_mode"] == "final"
    total = sum(len(method_specs_by_outcome[outcome]) for outcome in outcomes) * iterations
    matrix: list[dict[str, Any]] = []
    completed_payloads: list[dict[str, Any]] = []
    ordinal = 0
    runs_root = run / "machine-learning" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    planned_rows = []
    for outcome in outcomes:
        outcome_task = outcome_tasks.get(outcome, config["task"])
        for spec in method_specs_by_outcome[outcome]:
            model_id = str(spec["id"])
            for iteration in range(1, iterations + 1):
                planned_rows.append({
                    "outcome": outcome,
                    "model_id": model_id,
                    "iteration": iteration,
                    "seed": int(ml.get("seed", 42)) + iteration - 1,
                    "task": outcome_task,
                    "implementation": spec.get("implementation", ""),
                    "implementation_status": spec.get("implementation_status", ""),
                    "runtime_available": spec.get("runtime_available", True),
                    "unavailable_reason": spec.get("unavailable_reason", ""),
                    "parameters": parameters_for(outcome_task, model_id),
                    "data_protocol": "eda_sequence" if model_id in WINDOW_REGRESSION_MODELS else "converted_table",
                })
    write_json(run / "machine-learning" / "planned-runs.json", {
        "status": "planned",
        "total_slots": total,
        "outcomes": outcomes,
        "methods": sorted({str(spec.get("id")) for specs in method_specs_by_outcome.values() for spec in specs}),
        "methods_by_outcome": {
            outcome: [spec.get("id") for spec in method_specs_by_outcome[outcome]] for outcome in outcomes
        },
        "outcome_tasks": outcome_tasks,
        "iterations": iterations,
        "evaluation_mode": ml["evaluation_mode"],
        "rows": planned_rows,
    })
    emit(callback, "机器学习计划", 1, 1, f"{len(outcomes)} 个 outcome × 任务分流 × {iterations} 轮 = {total} 个槽位")

    for outcome in outcomes:
        source_path = Path(ml_data["outcome_paths"][outcome])
        time_column = ml.get("time_column")
        time_available = bool(time_column and time_column in _read_text_csv_header(source_path))
        outcome_task = outcome_tasks.get(outcome, config["task"])
        outcome_group_column = outcome_group_columns.get(outcome, config["group_column"])
        for spec in method_specs_by_outcome[outcome]:
            model_id = str(spec.get("id"))
            uses_sequence = model_id in WINDOW_REGRESSION_MODELS
            for iteration in range(1, iterations + 1):
                ordinal += 1
                if cancel is not None and cancel.is_set():
                    raise PipelineCancelled("用户取消了多因变量、多方法机器学习")
                iteration_dir = runs_root / outcome / model_id / f"iteration-{iteration:02d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                seed = int(ml.get("seed", 42)) + iteration - 1
                split_strategy = "sequence_team" if uses_sequence else (
                    "time" if outcome_task == "regression" and model_id.startswith("T") else "group"
                )
                row: dict[str, Any] = {
                    "outcome": outcome,
                    "model_id": model_id,
                    "iteration": iteration,
                    "seed": seed,
                    "task": outcome_task,
                    "status": "pending",
                    "split_strategy": split_strategy,
                    "run_directory": str(iteration_dir),
                }
                emit(callback, "机器学习迭代", ordinal - 1, total, f"{outcome} · {model_id} · 第 {iteration}/{iterations} 轮")
                if split_strategy == "time" and not time_available:
                    row["status"] = "unavailable"
                    row["reason"] = "预测模型需要显式配置并提供可核验的时间列；表格行序不可用于预测"
                    write_json(iteration_dir / "status.json", row)
                    matrix.append(row)
                    continue
                if (spec.get("implementation_status") != "available" or spec.get("runtime_available") is False
                        or (uses_sequence and outcome not in sequence_data)):
                    row["status"] = "unavailable"
                    row["reason"] = spec.get("unavailable_reason") or (
                        sequence_error or "窗口回归需要 eda_sequence 及已核验的逐记录回答序列"
                        if uses_sequence and outcome not in sequence_data else "模型在当前 PYML 目录或运行环境中不可用"
                    )
                    write_json(iteration_dir / "status.json", row)
                    matrix.append(row)
                    emit(callback, "机器学习迭代", ordinal, total, f"{outcome} · {model_id} · 不可用")
                    continue
                try:
                    split_kwargs: dict[str, Any] = {"seed": seed, "strategy": split_strategy}
                    sequence_config = None
                    iteration_source = source_path
                    if split_strategy == "group":
                        split_kwargs["group_column"] = outcome_group_column
                    elif uses_sequence:
                        split_kwargs["strategy"] = "holdout"
                        iteration_source = Path(sequence_data[outcome]["paths"][iteration])
                        sequence_config = SequenceConfig(
                            group_column="meta__sequence_id", order_mode="row",
                            window=ml["eda_sequence"]["window"], horizon=ml["eda_sequence"]["horizon"],
                        )
                    else:
                        split_kwargs["time_column"] = time_column
                    experiment = ExperimentConfig(
                        dataset=DatasetConfig(
                            source_path=str(iteration_source),
                            target_column=outcome,
                            feature_columns=tuple(ml_data["features"]),
                        ),
                        task=outcome_task,
                        model_id=model_id,
                        parameters=parameters_for(outcome_task, model_id),
                        split=SplitConfig(**split_kwargs),
                        sequence=sequence_config,
                        output_dir=str(iteration_dir / "artifacts"),
                    )
                    experiment.validate()
                    sequence_audit = None
                    if uses_sequence:
                        snapshot, plan = build_extended_snapshot(experiment)
                        structure = _read_text_csv(iteration_source)[["meta__group_id", "meta__subject_id"]]
                        teams_seen, subjects_seen = set(), set()
                        sequence_audit = {}
                        for split_name, partition in plan.partitions.items():
                            rows = structure.iloc[partition.row_positions]
                            teams, subjects = set(rows["meta__group_id"]), set(rows["meta__subject_id"])
                            if teams_seen & teams or subjects_seen & subjects:
                                raise PipelineError("原生 SequencePlan 将同一团队或被试分到了多个分区")
                            teams_seen.update(teams)
                            subjects_seen.update(subjects)
                            sequence_audit[split_name] = {
                                "team_count": len(teams), "subject_count": len(subjects),
                                "source_rows": len(rows), "windows": len(partition.window_target_positions),
                            }
                        session = prepare_experiment(experiment, snapshot=snapshot, sequence_plan=plan)
                    else:
                        session = prepare_experiment(experiment)
                    validation = {
                        "metrics": session.metrics,
                        "config": experiment.to_dict(),
                        "test_evaluation_count": session.test_evaluation_count,
                    }
                    write_json(iteration_dir / "validation.json", validation)
                    metrics = session.metrics
                    recorded_rows = pd.DataFrame(session.rows)
                    audit = {"split_counts": {name: len(positions) for name, positions in session.splits.items()},
                             "test_evaluation_count": 0}
                    artifact_paths = {}
                    if final_test:
                        freeze_experiment(session, expected_config=experiment)
                        write_json(iteration_dir / "frozen-config.json", experiment.to_dict())
                        result = evaluate_test(session)
                        metrics, recorded_rows, audit = result.metrics, result.results, result.audit
                        artifact_paths = result.artifact_paths
                    recorded_rows.to_csv(iteration_dir / "results.csv", index=False, encoding="utf-8-sig")
                    payload = {
                        "status": "completed", "outcome": outcome, "model_id": model_id,
                        "iteration": iteration, "seed": seed, "config": experiment.to_dict(),
                        "metrics": metrics, "audit": audit, "artifact_paths": artifact_paths,
                        "evaluation_mode": ml["evaluation_mode"],
                        "all_missing_features": ml_data["prep"]["all_missing_features"],
                        "test_evaluation_count": session.test_evaluation_count,
                    }
                    if sequence_audit is not None:
                        payload["sequence_split_audit"] = sequence_audit
                    write_json(iteration_dir / "result.json", payload)
                    row.update({
                        "status": "completed",
                        "metrics": metrics,
                        "audit": audit,
                        "result_path": str(iteration_dir / "result.json"),
                    })
                    completed_payloads.append(payload)
                except PipelineCancelled:
                    raise
                except Exception as exc:
                    row.update({
                        "status": "failed",
                        "reason": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    write_json(iteration_dir / "error.json", row)
                matrix.append(row)
                emit(callback, "机器学习迭代", ordinal, total, f"{outcome} · {model_id} · {row['status']}")

    fieldnames = ["outcome", "model_id", "iteration", "seed", "task", "status", "split_strategy", "reason", "run_directory", "result_path"]
    with (run / "machine-learning" / "method-matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matrix)
    counts: dict[str, int] = {}
    for row in matrix:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    primary = next(
        (payload for payload in completed_payloads
         if payload["outcome"] == config["target_column"] and payload["model_id"] == config["model_id"] and payload["iteration"] == 1),
        None,
    )
    ml_status = summarize_slot_status(counts, total)
    all_results = {
        "status": ml_status,
        "outcomes": outcomes,
        "methods": sorted({str(spec.get("id")) for specs in method_specs_by_outcome.values() for spec in specs}),
        "methods_by_outcome": {
            outcome: [spec.get("id") for spec in method_specs_by_outcome[outcome]] for outcome in outcomes
        },
        "outcome_tasks": outcome_tasks,
        "iterations": iterations,
        "evaluation_mode": ml["evaluation_mode"],
        "counts": counts,
        "matrix": matrix,
        "primary_result": primary,
        "all_missing_features": ml_data["prep"]["all_missing_features"],
    }
    write_json(run / "machine-learning" / "all-results.json", all_results)
    write_json(run / "machine-learning" / "result.json", primary or all_results)
    save_state(run, "ml_complete", ml_status=ml_status, all_results_path=str(run / "machine-learning" / "all-results.json"))
    return all_results


def run_report(
    config: dict[str, Any],
    run: Path,
    callback: ProgressCallback | None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Build the APA report from the completed run without changing ML results."""
    report_template = CORE_DIR / "make_report.py"
    if not report_template.is_file():
        return {"status": "unavailable", "reason": f"报告生成器不存在：{report_template}"}
    report_dir = run / "reports" / datetime.now().strftime("%Y%m%d")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_script = report_dir / "make_report.py"
    shutil.copy2(report_template, report_script)
    log_path = report_dir / "report.log"
    command = [
        str(_console_python(Path(sys.executable))), "-X", "utf8", str(report_script),
        "--run", str(run), "--source", str(config["input"]), "--out", str(report_dir),
    ]
    emit(callback, "生成 APA 报告", 0, 1, str(report_dir))
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=str(report_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            started = time.monotonic()
            last_emit = started
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    _stop_process(process)
                    raise PipelineCancelled("用户取消了 APA 报告生成")
                current = time.monotonic()
                if current - last_emit >= 5:
                    elapsed = int(current - started)
                    emit(callback, "生成 APA 报告", 0, 1, f"报告生成进行中，已用 {elapsed} 秒")
                    last_emit = current
                time.sleep(0.5)
            return_code = process.returncode
        check_path = report_dir / "report-check.json"
        if return_code != 0 or not check_path.is_file():
            return {"status": "failed", "reason": f"APA 报告生成失败，详见 {log_path}", "log": str(log_path)}
        check = read_json(check_path)
        emit(callback, "生成 APA 报告", 1, 1, str(report_dir))
        return {"status": "completed", "directory": str(report_dir), "check": str(check_path), "log": str(log_path), "citation_status": check.get("status")}
    except PipelineCancelled:
        raise
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "log": str(log_path)}


def write_integrity(run: Path) -> dict[str, Any]:
    files = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "integrity.json":
            files.append({"path": str(path.relative_to(run)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size})
    payload = {"schema_version": 1, "generated_at": now(), "files": files}
    write_json(run / "integrity.json", payload)
    return payload


def run_pipeline(config: dict[str, Any], *, progress_callback: ProgressCallback | None = None, cancel_event: threading.Event | None = None) -> dict[str, Any]:
    config = validate_config(config)
    run = make_run_dir(Path(config["output_root"]))
    write_json(run / "config.json", config)
    save_state(run, "started", input=config["input"], run_directory=str(run))
    try:
        dependencies = check_runtime_dependencies(config, run, progress_callback, cancel_event)
        save_state(run, "dependencies_checked", dependency_status=dependencies)
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("用户取消了依赖检查")
        conversion = run_conversion(config, run, progress_callback, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("用户取消了转换与机器学习")
        ml_data = prepare_ml_dataset(config, run, progress_callback)
        save_state(run, "ml_dataset_ready", conversion_summary=conversion, data_preparation=ml_data["prep"])
        result = run_machine_learning(config, run, ml_data, progress_callback, cancel_event)
        summary = {
            "status": result.get("status", "failed"), "run_directory": str(run), "conversion": conversion,
            "machine_learning": result, "data_preparation": ml_data["prep"],
            "dependencies": dependencies,
            "report": {"status": "pending"},
            "integrity_file_count": None,
        }
        write_json(run / "oneclick-summary.json", summary)
        report = run_report(config, run, progress_callback, cancel_event)
        summary["report"] = report
        if report.get("status") != "completed":
            summary["status"] = ("completed_with_report_failure" if summary["status"] == "completed"
                                 else f"{summary['status']}_report_failure")
        write_json(run / "oneclick-summary.json", summary)
        summary["integrity_file_count"] = sum(
            path.is_file() and path.name != "integrity.json" for path in run.rglob("*")
        )
        write_json(run / "oneclick-summary.json", summary)
        save_state(run, summary["status"], summary_path=str(run / "oneclick-summary.json"))
        write_integrity(run)
        emit(progress_callback, "完成", 1, 1, str(run))
        return summary
    except PipelineCancelled as exc:
        save_state(run, "cancelled", error=str(exc))
        raise
    except Exception as exc:
        error = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "run_directory": str(run)}
        write_json(run / "error.json", error)
        summary_path = run / "oneclick-summary.json"
        if summary_path.is_file():
            failed_summary = read_json(summary_path)
            failed_summary.update(status="failed", integrity_file_count=None, error=str(exc))
            write_json(summary_path, failed_summary)
        save_state(run, "failed", error_path=str(run / "error.json"))
        raise


def validate_only(config: dict[str, Any]) -> dict[str, Any]:
    config = validate_config(config)
    ensure_pyml_importable(config)
    return {"status": "ready_for_pilot", "config": config, "conversion_and_training": "NOT RUN"}


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path)


def progress_printer(stage: str, current: int, total: int, detail: str) -> None:
    print(f"[{stage}] {current}/{total} {detail}", flush=True)


def run_gui(initial_config: dict[str, Any] | None = None, smoke: bool = False) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    app = tk.Tk()
    app.title("PYEDA → PYML 一站式转换与机器学习")
    app.geometry("900x680")
    app.minsize(760, 560)
    initial_config = initial_config or {}
    initial_ml = dict(initial_config.get("machine_learning") or {})
    values = {key: tk.StringVar() for key in ("plugin", "input", "output", "target", "group", "model")}
    values["plugin"].set(str(initial_config.get("pyeda_plugin", "")))
    values["input"].set(str(initial_config.get("input", "")))
    values["output"].set(str(initial_config.get("output_root", "")))
    values["target"].set(str(initial_config.get("target_column", "")))
    values["group"].set(str(initial_config.get("group_column", "")))
    values["model"].set(str(initial_config.get("model_id", "R01")))
    values["task"] = tk.StringVar(value=str(initial_config.get("task", "regression")))
    configured_outcomes = list(initial_ml.get("outcomes") or [])
    configured_tasks = dict(initial_ml.get("outcome_tasks") or {})
    raw_method_policy = initial_ml.get("methods", "all_regression")
    method_policy = (
        f"回归 {raw_method_policy.get('regression', 'all_regression')}；分类 {raw_method_policy.get('classification', 'all_classification')}"
        if isinstance(raw_method_policy, dict) else str(raw_method_policy)
    )
    iteration_count = int(initial_ml.get("iterations", 3))
    task_scope = "；".join(f"{str(outcome).replace('target__', '')}={task}" for outcome, task in configured_tasks.items()) or "未配置，运行前需确认"
    scope = tk.StringVar(value=f"预设范围：{len(configured_outcomes) or '全部'} 个 outcome · 任务映射 {task_scope} · 方法策略 {method_policy} · {iteration_count} 轮迭代 · 全特征输入")
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancel = threading.Event()
    worker: threading.Thread | None = None

    root = ttk.Frame(app, padding=16)
    root.pack(fill="both", expand=True)
    ttk.Label(root, text="PYEDA 结果目录转换后按显式目标和主体分组运行 PYML", font=("Microsoft YaHei UI", 14)).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

    fields = [("plugin", "PYEDA 插件目录", True), ("input", "PYEDA 结果目录", True), ("output", "输出根目录", True), ("target", "目标列", False), ("group", "分组列", False), ("model", "模型 ID", False)]
    for row, (key, label, directory) in enumerate(fields, 1):
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(root, textvariable=values[key], width=70).grid(row=row, column=1, sticky="ew", pady=5)
        if directory:
            ttk.Button(root, text="选择", command=lambda k=key: values[k].set(filedialog.askdirectory() or values[k].get())).grid(row=row, column=2, padx=(8, 0))
    ttk.Label(root, text="默认任务").grid(row=7, column=0, sticky="w", pady=5)
    ttk.Combobox(root, textvariable=values["task"], state="readonly", values=("regression", "classification"), width=20).grid(row=7, column=1, sticky="w", pady=5)
    ttk.Label(root, textvariable=scope, wraplength=820).grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 2))
    status = tk.StringVar(value="等待配置")
    ttk.Label(root, textvariable=status, wraplength=820).grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 6))
    timing = tk.StringVar(value="已用 00:00:00 · 剩余时间正在收集速度")
    ttk.Label(root, textvariable=timing).grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 5))
    bar = ttk.Progressbar(root, maximum=100)
    bar.grid(row=11, column=0, columnspan=3, sticky="ew")
    log = tk.Text(root, height=18, state="disabled", wrap="word")
    log.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    root.columnconfigure(1, weight=1)
    root.rowconfigure(12, weight=1)
    start = ttk.Button(root, text="开始一站式运行")
    start.grid(row=13, column=1, sticky="e", pady=(12, 0))
    stop = ttk.Button(root, text="取消", state="disabled")
    stop.grid(row=13, column=2, sticky="e", pady=(12, 0))
    started_at = 0.0
    last_stage = ""
    samples: list[tuple[float, int]] = []

    def append(text: str) -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def build_config() -> dict[str, Any]:
        return {
            "pyeda_plugin": values["plugin"].get().strip(), "input": values["input"].get().strip(),
            "output_root": values["output"].get().strip(), "target_column": values["target"].get().strip(),
            "group_column": values["group"].get().strip(), "task": values["task"].get(), "model_id": values["model"].get().strip(),
            "machine_learning": {
                "seed": int(initial_ml.get("seed", 42)),
                "parameters": dict(initial_ml.get("parameters") or {}),
                "parameters_by_task": dict(initial_ml.get("parameters_by_task") or {}),
                "parameters_by_model": dict(initial_ml.get("parameters_by_model") or {}),
                "eda_sequence": initial_ml.get("eda_sequence"),
                "feature_columns": initial_ml.get("feature_columns"),
                "outcomes": list(initial_ml.get("outcomes")) if initial_ml.get("outcomes") else None,
                "outcome_tasks": dict(initial_ml.get("outcome_tasks") or {}),
                "outcome_group_columns": dict(initial_ml.get("outcome_group_columns") or {}),
                "ioc_classification": dict(initial_ml.get("ioc_classification") or {}),
                "methods": initial_ml.get("methods") or [values["model"].get().strip()],
                "iterations": int(initial_ml.get("iterations", 3)),
                "all_missing_feature_policy": initial_ml.get("all_missing_feature_policy", "constant_zero_with_indicator"),
                "time_column": initial_ml.get("time_column"),
                "evaluation_mode": initial_ml.get("evaluation_mode", "validation"),
                "confirm_final_holdout": initial_ml.get("confirm_final_holdout", False),
            },
            "conversion": dict(initial_config.get("conversion") or {"source": "participant_task_role", "include_outcomes": True, "only_valid_records": True, "verify_manifest_hashes": True}),
            "pyml_root": str(initial_config.get("pyml_root") or find_pyml_root() or ""),
            "python": str(initial_config.get("python") or sys.executable),
            "study": dict(initial_config.get("study") or {}),
        }

    def run() -> None:
        nonlocal worker, started_at, last_stage, samples
        missing_task_outcomes = [
            outcome for outcome in configured_outcomes
            if outcome not in configured_tasks and outcome != "target__ioc_binary"
        ]
        if missing_task_outcomes:
            labels = "、".join(str(outcome).replace("target__", "") for outcome in missing_task_outcomes)
            choice = messagebox.askyesnocancel(
                "确认任务类型",
                f"以下 outcome 尚未指定任务类型：{labels}\n\n选择“是”按分类处理，选择“否”按预测/回归处理。混合任务请先在配置文件中填写 outcome_tasks。",
                parent=app,
            )
            if choice is None:
                return
            selected_task = "classification" if choice else "regression"
            initial_ml.setdefault("outcome_tasks", {}).update({outcome: selected_task for outcome in missing_task_outcomes})
            updated_scope = "；".join(
                f"{str(outcome).replace('target__', '')}={task}"
                for outcome, task in dict(initial_ml.get("outcome_tasks") or {}).items()
            )
            scope.set(f"预设范围：{len(configured_outcomes) or '全部'} 个 outcome · 任务映射 {updated_scope} · 方法策略 {method_policy} · {iteration_count} 轮迭代 · 全特征输入")
        try:
            config = validate_config(build_config())
        except Exception as exc:
            messagebox.showerror("配置检查失败", str(exc), parent=app)
            return
        cancel.clear()
        started_at = time.monotonic()
        last_stage = ""
        samples = []
        start.configure(state="disabled")
        stop.configure(state="normal")
        append("已通过输入契约检查，开始转换")
        def callback(stage: str, current: int, total: int, detail: str) -> None:
            events.put(("progress", (stage, current, total, detail)))
        def task() -> None:
            try:
                result = run_pipeline(config, progress_callback=callback, cancel_event=cancel)
            except Exception as exc:
                events.put(("error", exc))
            else:
                events.put(("done", result))
        worker = threading.Thread(target=task, daemon=True)
        worker.start()
        poll()

    def poll() -> None:
        nonlocal last_stage, samples
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "progress":
                    stage, current, total, detail = payload
                    status.set(f"{stage}：{detail}")
                    bar["value"] = 100 * current / max(total, 1)
                    elapsed = int(time.monotonic() - started_at) if started_at else 0
                    if stage != last_stage:
                        last_stage = stage
                        samples = []
                    if not samples or current > samples[-1][1]:
                        samples.append((time.monotonic(), current))
                    eta = "剩余时间正在收集速度"
                    if len(samples) >= 3:
                        elapsed_sample = samples[-1][0] - samples[0][0]
                        completed_sample = samples[-1][1] - samples[0][1]
                        if elapsed_sample > 0 and completed_sample > 0:
                            remaining = max(0, int((total - current) * elapsed_sample / completed_sample))
                            eta = f"剩余时间约 {remaining // 60:02d}:{remaining % 60:02d}（估算）"
                    timing.set(f"已用 {elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d} · {eta}")
                    append(f"{stage} {current}/{total} {detail}")
                elif kind == "done":
                    start.configure(state="normal")
                    stop.configure(state="disabled")
                    final_status = str(payload.get("status", "completed"))
                    status.set(f"终态：{final_status}")
                    elapsed = int(time.monotonic() - started_at) if started_at else 0
                    timing.set(f"已用 {elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d} · 结果已写入")
                    append(f"输出目录：{payload['run_directory']}")
                    messagebox.showinfo("运行终态", f"{final_status}\n{payload['run_directory']}", parent=app)
                elif kind == "error":
                    start.configure(state="normal")
                    stop.configure(state="disabled")
                    status.set("运行失败，错误详情已保存")
                    append(str(payload))
                    messagebox.showerror("运行失败", str(payload), parent=app)
        except queue.Empty:
            pass
        if worker is not None and worker.is_alive():
            app.after(150, poll)

    start.configure(command=run)
    stop.configure(command=cancel.set)
    if smoke:
        app.after(700, app.destroy)
    app.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PYEDA 转换与 PYML 机器学习一站式入口")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="使用配置打开预填充 GUI")
    parser.add_argument("--no-open", action="store_true", help="保留参数以兼容启动器")
    parser.add_argument("--smoke", action="store_true", help="启动 GUI 后自动退出，用于入口验收")
    args = parser.parse_args(argv)
    if args.gui:
        return run_gui(load_config(args.config) if args.config else None, args.smoke)
    if args.config is None:
        return run_gui(None, args.smoke)
    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps(validate_only(config), ensure_ascii=False, indent=2, default=json_default))
        return 0
    result = run_pipeline(config, progress_callback=progress_printer)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)
