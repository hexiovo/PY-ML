"""Run an unsupervised t-SNE example and show its held-out capability limit."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyml_workbench import DatasetConfig, ExperimentConfig, evaluate_test, freeze_experiment, prepare_experiment


def main() -> None:
    rng = np.random.default_rng(23)
    centers = np.array([[-2.0, -1.0], [2.0, 1.0], [-2.0, 2.0]])
    groups = [center + rng.normal(0.0, 0.3, size=(24, 2)) for center in centers]
    values = np.vstack(groups)
    frame = pd.DataFrame(values, columns=["signal_a", "signal_b"])

    with tempfile.TemporaryDirectory(prefix="pyml-workbench-unsupervised-") as temporary:
        root = Path(temporary)
        source = root / "synthetic-unlabeled.csv"
        frame.to_csv(source, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(source),
                feature_columns=("signal_a", "signal_b"),
            ),
            task="dimensionality reduction",
            model_id="D10",
            parameters={"n_components": 2, "perplexity": 8.0, "random_state": 23},
            output_dir=str(root / "artifacts"),
        )

        session = prepare_experiment(config)
        print("training trustworthiness:", session.metrics["train"])
        print("validation:", session.metrics["validation"])
        assert session.metrics["validation"]["status"] == "not_supported"
        assert session.metrics["validation"]["reason"].find("no transform") >= 0

        freeze_experiment(session, expected_config=config)
        result = evaluate_test(session)
        print("test:", result.metrics["test"])
        print("test evaluations:", result.audit["test_evaluation_count"])
        print("held-out embedding:", "not available; fitted t-SNE has no transform method")


if __name__ == "__main__":
    main()
