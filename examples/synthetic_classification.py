"""Run a small, deterministic classification example without user data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    evaluate_test,
    freeze_experiment,
    load_model,
    predict,
    prepare_experiment,
)


def main() -> None:
    values = np.linspace(-2.0, 2.0, 80)
    noise = np.random.default_rng(42).normal(0.0, 0.15, size=len(values))
    frame = pd.DataFrame(
        {
            "signal": values + noise,
            "target": (values > 0).astype("int64"),
        }
    )

    with tempfile.TemporaryDirectory(prefix="pyml-workbench-example-") as temporary:
        root = Path(temporary)
        source = root / "synthetic.csv"
        output = root / "artifacts"
        frame.to_csv(source, index=False)

        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(source),
                target_column="target",
                feature_columns=("signal",),
            ),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 500},
            output_dir=str(output),
        )

        session = prepare_experiment(config)
        print("validation:", session.metrics["validation"])
        assert session.test_evaluation_count == 0

        freeze_experiment(session, expected_config=config)
        result = evaluate_test(session)
        print("test:", result.metrics["test"])
        print("test evaluations:", result.audit["test_evaluation_count"])

        model = load_model(result.artifact_paths["model"])
        if model.capabilities.get("predict"):
            prediction = predict(model, frame.loc[:4, ["signal"]])
            print("first five predictions:", prediction.tolist())
        else:
            print("loaded model does not expose predict for new rows")

        print("artifacts:", sorted(path.name for path in output.iterdir()))


if __name__ == "__main__":
    main()
