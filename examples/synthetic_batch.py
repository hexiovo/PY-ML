"""使用明确标注的模拟分类数据，演示批量搜索、冻结 winner、最终评估与缓存恢复。

默认把 CSV、SQLite 历史和模型会话写入系统临时目录，并在退出时清理。
传入 --output DIR 可将完整演示产物保留在一个新建或空目录中。
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    HistoryStore,
    ObjectiveSpec,
    SearchSpace,
    SearchSpec,
    SplitConfig,
    create_search_jobs,
    finalize_frozen_search,
    freeze_search_winner,
    get_job_summary,
    load_search_result,
    run_search_jobs,
)


MODELS = {
    "C01": ("C", [0.25, 1.0]),
    "C02": ("alpha", [0.25, 1.0]),
}


def _write_synthetic_csv(path: Path, *, seed: int, offset: float) -> None:
    """Write a small deterministic CSV whose rows are explicitly synthetic."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1], dtype=int), 40)
    rng.shuffle(labels)
    signed = labels.astype(float) * 2.0 - 1.0
    noise = rng.normal(size=(len(labels), 3))
    frame = pd.DataFrame(
        {
            "模拟特征_1": signed + noise[:, 0] * 0.9 + offset,
            "模拟特征_2": -0.7 * signed + noise[:, 1] + offset * 0.2,
            "模拟特征_3": 0.4 * signed + noise[:, 2] * 1.1,
            "模拟类别": labels,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_demo(root: Path) -> list[dict[str, object]]:
    data_dir = root / "模拟数据"
    batch_dir = root / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = [
        data_dir / "模拟分类集_A.csv",
        data_dir / "模拟分类集_B.csv",
    ]
    _write_synthetic_csv(dataset_paths[0], seed=2026, offset=0.0)
    _write_synthetic_csv(dataset_paths[1], seed=2027, offset=0.35)

    requests = []
    for data_index, dataset_path in enumerate(dataset_paths):
        for model_id, (parameter_name, values) in MODELS.items():
            config = ExperimentConfig(
                dataset=DatasetConfig(
                    source_path=str(dataset_path),
                    target_column="模拟类别",
                    feature_columns=("模拟特征_1", "模拟特征_2", "模拟特征_3"),
                ),
                task="classification",
                model_id=model_id,
                split=SplitConfig(seed=42),
            )
            space = SearchSpace.from_dict(
                {
                    "fields": {
                        parameter_name: {
                            "type": "real",
                            "low": 0.1,
                            "high": 2.0,
                            "values": values,
                        }
                    }
                }
            )
            spec = SearchSpec(
                method="grid",
                space=space,
                objective=ObjectiveSpec(),
                max_fits=2,
                timeout_seconds=1200.0,
                max_proposals=10,
                seed=70 + data_index,
            )
            requests.append((config, spec))

    history_path = batch_dir / "history.sqlite3"
    artifact_root = batch_dir / "artifacts"
    jobs = create_search_jobs(history_path, artifact_root, requests)
    job_ids = [str(job["job_id"]) for job in jobs]
    outcomes = run_search_jobs(history_path, job_ids, max_workers=1)
    failed = [item for item in outcomes if item.get("status") != "completed"]
    if failed:
        raise RuntimeError(f"模拟批次中的搜索未全部完成：{failed}")

    store = HistoryStore(history_path)
    rows: list[dict[str, object]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        result = load_search_result(history_path, job_id)
        if result is None or result.winner is None:
            raise RuntimeError(f"任务 {job_id} 没有可冻结的验证集 winner")
        if result.actual_fit_count > 2:
            raise RuntimeError(f"任务 {job_id} 超过示例设定的 2 次搜索拟合")

        frozen = freeze_search_winner(history_path, job_id)
        final = finalize_frozen_search(history_path, job_id)
        if final.session.test_evaluation_count != 1:
            raise RuntimeError(
                f"任务 {job_id} 最终 test 评估次数应为 1，实际为 "
                f"{final.session.test_evaluation_count}"
            )

        # 第二次读取同一已完成任务应从最终会话缓存恢复，不再重拟合或评估 test。
        restored_final = finalize_frozen_search(history_path, job_id)
        if (
            not restored_final.cached
            or restored_final.final_run_id != final.final_run_id
            or restored_final.session.test_evaluation_count != 1
        ):
            raise RuntimeError(f"任务 {job_id} 未能从最终会话缓存安全恢复")

        reloaded_search = load_search_result(history_path, job_id)
        if reloaded_search is None or reloaded_search.fingerprint != result.fingerprint:
            raise RuntimeError(f"任务 {job_id} 的搜索历史恢复指纹不一致")
        permission = store.get_test_permission(job_id)
        if not permission or permission["state"] != "completed":
            raise RuntimeError(f"任务 {job_id} 的一次性测试许可未持久化完成")

        summary = get_job_summary(history_path, job_id)
        config = result.config.to_dict()
        final_metrics = restored_final.session.metrics.get("test", {})
        rows.append(
            {
                "job_id": job_id,
                "dataset": Path(config["dataset"]["source_path"]).name,
                "model_id": config["model_id"],
                "status": summary["status"],
                "search_fits": summary["actual_fit_count"],
                "proposals": summary["proposal_count"],
                "winner_trial_id": frozen.winner_trial_id,
                "selection_validation": frozen.metadata["validation_metrics"],
                "final_test": final_metrics,
                "test_evaluation_count": restored_final.session.test_evaluation_count,
                "final_cache_reload": restored_final.cached,
                "search_fingerprint": reloaded_search.fingerprint,
            }
        )

    all_jobs = store.list_jobs()
    if len(all_jobs) != 4:
        raise RuntimeError(f"应有两份模拟数据 × 两个模型共 4 个历史任务，实际 {len(all_jobs)}")
    summary_path = root / "演示结果.json"
    summary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"SQLite 历史：{history_path}")
    print(f"搜索会话与冻结/最终会话：{artifact_root}")
    print(f"演示结果：{summary_path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行一组明确标记的模拟分类数据批量搜索演示。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="保留演示 CSV、SQLite 历史和会话产物的空目录；省略时使用自动清理的临时目录。",
    )
    args = parser.parse_args()

    if args.output is not None:
        root = args.output.expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise SystemExit(f"输出路径不是目录：{root}")
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"为避免覆盖现有文件，--output 必须指向新建或空目录：{root}")
        root.mkdir(parents=True, exist_ok=True)
        run_demo(root)
        print(f"演示产物已保留：{root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="pyml-synthetic-batch-") as temporary:
        root = Path(temporary)
        print("数据标记：以下两份 CSV 全部由本示例生成，仅用于演示，不是项目原始数据。")
        print(f"临时演示目录：{root}（程序结束时自动清理）")
        run_demo(root)
        print("临时演示产物会在程序结束时自动清理；若需保留，请使用 --output 空目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
