"""Core search, explicit selection, refit, test-once, and export integration for S03."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.experiment import load_model
from pyml_workbench.search import search
from pyml_workbench.selection import export_selected, finalize_selected, freeze_selection
from pyml_workbench.sequence import SequenceConfig


def _frame() -> pd.DataFrame:
    rows = []
    for group in range(15):
        for step in range(12):
            value = float(group * 12 + step)
            rows.append({
                "group": f"g{group:02d}",
                "time": step,
                "value": value,
                "target": float(group * 0.5 + step * 0.25),
            })
    return pd.DataFrame(rows)


class ExtendedSelectionTests(unittest.TestCase):
    def test_hmm_and_window_deep_winners_refit_test_once_cache_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sequence.csv"
            _frame().to_csv(source, index=False)

            cases = (
                (
                    "H01",
                    ExperimentConfig(
                        dataset=DatasetConfig(str(source), feature_columns=("value",)),
                        task="sequence_modeling",
                        model_id="H01",
                        parameters={"n_components": 2, "n_iter": 2},
                        sequence=SequenceConfig(
                            group_column="group",
                            time_column="time",
                            order_mode="time",
                            observation_columns=("value",),
                        ),
                    ),
                ),
                (
                    "N06",
                    ExperimentConfig(
                        dataset=DatasetConfig(
                            str(source),
                            target_column="target",
                            feature_columns=("value",),
                        ),
                        task="regression",
                        model_id="N06",
                        parameters={
                            "hidden_size": 4,
                            "batch_size": 16,
                            "max_epochs": 2,
                            "patience": 1,
                            "random_state": 42,
                        },
                        sequence=SequenceConfig(
                            group_column="group",
                            time_column="time",
                            order_mode="time",
                            window=3,
                            horizon=1,
                        ),
                    ),
                ),
            )

            for model_id, config in cases:
                with self.subTest(model_id=model_id):
                    result = search(
                        config,
                        {"method": "grid", "space": {"fields": {}, "fixed": {}}, "max_fits": 1},
                        artifact_dir=root / f"search-{model_id}",
                    )
                    self.assertEqual(result.status, "complete")
                    self.assertEqual(result.actual_fit_count, 1)
                    self.assertIsNotNone(result.winner)
                    self.assertEqual(result.test_evaluation_count, 0)

                    selection = freeze_selection(result)
                    final_path = root / f"final-{model_id}.joblib"
                    final = finalize_selected(selection, session_path=final_path)
                    self.assertEqual(final.state, "tested")
                    self.assertEqual(final.session.test_evaluation_count, 1)
                    sample_key = "n_observations" if model_id == "H01" else "sample_count"
                    self.assertGreater(final.session.metrics["test"][sample_key], 0)

                    cached = finalize_selected(selection, session_path=final_path)
                    self.assertTrue(cached.cached)
                    self.assertEqual(cached.session.test_evaluation_count, 1)
                    self.assertEqual(cached.session.metrics["test"], final.session.metrics["test"])

                    paths = export_selected(final, root / f"export-{model_id}")
                    model = load_model(paths["model"])
                    self.assertEqual(model.model_id, model_id)
                    if model_id == "H01":
                        plan = result.sequence_plan
                        inputs = {
                            "observations": plan.validation.observations[:4],
                            "lengths": [4],
                        }
                    else:
                        inputs = result.sequence_plan.validation.window_inputs[:4]
                    self.assertEqual(len(model.predict(inputs)), 4)


if __name__ == "__main__":
    unittest.main()
