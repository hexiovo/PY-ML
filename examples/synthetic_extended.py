"""Run small CPU-only HMM and GRU workflows on generated temporary data."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    SplitConfig,
    finalize_experiment,
    freeze_experiment,
    load_model,
    prepare_experiment,
)
from pyml_workbench.extended_experiment import predict_extended
from pyml_workbench.sequence import SequenceConfig


def _make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20261001)
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    for group in range(20):
        phase = group * 0.025
        for step in range(30):
            signal = float(np.sin(step / 4.0 + phase) + group * 0.006 + rng.normal(0, 0.035))
            target = float(np.sin(step / 4.0 + phase) + group * 0.006 + rng.normal(0, 0.02))
            rows.append(
                {
                    "series": f"series-{group:02d}",
                    "timestamp": origin + pd.Timedelta(hours=step),
                    "signal": signal,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"{name}={json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=_json_default)}")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _report_sequence_counts(session, *, windows: bool) -> None:
    plan = session.sequence_plan
    rows = {name: len(part.row_positions) for name, part in plan.partitions.items()}
    groups = {name: len(part.group_ids) for name, part in plan.partitions.items()}
    report = {"groups": groups, "rows": rows}
    if windows:
        report["windows"] = {name: len(part.window_targets) for name, part in plan.partitions.items()}
        print(f"N06 window={plan.sequence_config.window} horizon={plan.sequence_config.horizon}")
    else:
        report["hmm_lengths"] = {
            name: {"sequence_count": len(part.lengths), "observations": int(part.lengths.sum())}
            for name, part in plan.partitions.items()
        }
    print(f"{session.config.model_id} split_counts={json.dumps(report, sort_keys=True)}")


def _finalize_preselected_session(config: ExperimentConfig, *, windows: bool):
    # The model and its parameters are chosen before the final test call.
    session = prepare_experiment(config)
    print(f"{config.model_id} validation_metrics:")
    _print_metrics("validation", session.metrics.get("validation", {}))
    _report_sequence_counts(session, windows=windows)
    if config.model_id == "N06":
        metadata = session.extended_training_metadata
        print(
            "N06 early_stopping="
            + json.dumps(
                {
                    "selected_epochs": metadata.get("selected_epochs"),
                    "best_epoch": metadata.get("best_epoch"),
                    "best_weight_policy": metadata.get("best_weight_policy"),
                    "effective_training_seed": metadata.get("effective_training_seed"),
                },
                sort_keys=True,
            )
        )
    freeze_experiment(session, expected_config=config)
    result = finalize_experiment(session)
    _print_metrics(f"{config.model_id} final_test_metrics", result.metrics.get("test", {}))
    print(
        f"{config.model_id} test_evaluation_count="
        f"{result.audit['test_evaluation_count']} finalized={session.finalized}"
    )
    if result.audit["test_evaluation_count"] != 1:
        raise RuntimeError(f"{config.model_id} did not consume exactly one final test evaluation")
    for name, value in result.metrics.get("test", {}).items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(float(value)):
            raise RuntimeError(f"{config.model_id} produced a non-finite test metric {name}")
    return session, result


def main() -> None:
    frame = _make_frame()
    with tempfile.TemporaryDirectory(prefix="pyml-extended-") as temporary:
        root = Path(temporary)
        source = root / "synthetic-series.csv"
        frame.to_csv(source, index=False)

        hmm_config = ExperimentConfig(
            dataset=DatasetConfig(source_path=str(source), feature_columns=("signal",)),
            task="sequence_modeling",
            model_id="H01",
            parameters={"n_components": 2, "n_iter": 8},
            split=SplitConfig(seed=17),
            output_dir=str(root / "artifacts" / "hmm"),
            sequence=SequenceConfig(
                group_column="series",
                time_column="timestamp",
                order_mode="time",
                observation_columns=("signal",),
            ),
        )
        hmm_session, hmm_result = _finalize_preselected_session(hmm_config, windows=False)
        hmm_model = load_model(hmm_result.artifact_paths["model"])
        new_observations = np.asarray(
            [[np.sin(index / 4.0)] for index in range(12)], dtype=np.float32
        )
        states = predict_extended(
            hmm_model,
            {"observations": new_observations, "lengths": [len(new_observations)]},
        )
        print(f"H01 reloaded_state_shape={np.asarray(states).shape}")

        gru_config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(source),
                target_column="target",
                feature_columns=("signal",),
            ),
            task="regression",
            model_id="N06",
            parameters={
                "hidden_size": 8,
                "num_layers": 1,
                "batch_size": 32,
                "max_epochs": 5,
                "patience": 2,
            },
            split=SplitConfig(seed=17),
            output_dir=str(root / "artifacts" / "gru"),
            sequence=SequenceConfig(
                group_column="series",
                time_column="timestamp",
                order_mode="time",
                window=10,
                horizon=1,
            ),
        )
        gru_session, gru_result = _finalize_preselected_session(gru_config, windows=True)
        gru_model = load_model(gru_result.artifact_paths["model"])
        phase = np.asarray([0.1, 0.4], dtype=np.float32)[:, None, None]
        steps = np.arange(10, dtype=np.float32)[None, :, None]
        new_windows = np.sin(steps / 4.0 + phase).astype(np.float32)
        predictions = predict_extended(gru_model, new_windows)
        print(f"N06 reloaded_prediction_shape={np.asarray(predictions).shape}")

        print("Completed two independently configured CPU examples; no test metric selected a model.")


if __name__ == "__main__":
    main()
