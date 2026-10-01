from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

from pyml_workbench.plotting import (
    PlotColumn,
    PlotKind,
    PlotProvenance,
    PlotSequenceMap,
    PlotSource,
    PlotSpec,
    available_plot_specs,
    build_plot_payload,
    render_plot_payload,
)


def _column(column_id, label, values, dtype="object"):
    return PlotColumn(column_id, label, dtype, tuple(values))


def _source(columns=(), outputs=(), *, positions=None, kind="loaded_eda", partition="all", owner_id="loaded"):
    if positions is None:
        size = len((columns or outputs)[0].values) if (columns or outputs) else 0
        positions = tuple(range(size))
    return PlotSource(
        schema_version=1,
        source_kind=kind,
        source_id="source-1",
        owner_kind={"loaded_eda": "loaded_dataset", "session_cache": "session", "batch_cache": "batch_job"}[kind],
        owner_id=owner_id,
        partition=partition,
        row_positions=tuple(positions),
        columns=tuple(columns),
        outputs=tuple(outputs),
        partition_count=len(positions),
    )


def _spec(source, kind, *column_ids):
    return next(
        item for item in available_plot_specs(source)
        if item.kind == kind and item.column_ids == tuple(column_ids)
    )


class PlotPayloadTests(unittest.TestCase):
    def test_plot_source_copies_numpy_scalars_and_rejects_nested_mutability(self):
        column = _column("value", np.int64(7), [np.int64(1), np.float64(2.5)])
        self.assertIs(type(column.label), int)
        self.assertEqual(column.values, (1, 2.5))
        with self.assertRaises(TypeError):
            PlotColumn("bad", "bad", "object", ([1, 2],))
        with self.assertRaises(TypeError):
            PlotColumn("bad-label", ["nested"], "object", (1,))

    def test_plot_source_requires_row_position_alignment_and_same_hash_shape(self):
        with self.assertRaises(ValueError):
            _source(columns=[_column("x", "x", [1, 2])], positions=[0])
        with self.assertRaises(ValueError):
            PlotProvenance(source_sha256="0" * 63)
        with self.assertRaises(TypeError):
            PlotSequenceMap(1, source_row_positions=([1, 2],))

    def test_histogram_uses_twenty_bins_and_reports_finite_exclusions(self):
        source = _source(columns=[_column("value", 17, [0, 1, 2, float("nan"), float("inf")], "float64")])
        payload = build_plot_payload(source, _spec(source, PlotKind.HISTOGRAM, "value"))
        self.assertEqual(len(payload.x_values), 21)
        self.assertEqual(len(payload.y_values), 20)
        self.assertEqual(sum(payload.y_values), 3)
        self.assertEqual((payload.partition_count, payload.effective_count, payload.excluded_count), (5, 3, 2))
        self.assertEqual(payload.sampled_count, 0)

    def test_typed_categories_keep_integer_string_and_missing_distinct(self):
        source = _source(columns=[_column("kind", 3, [1, "1", 1, None, float("nan")])])
        payload = build_plot_payload(source, _spec(source, PlotKind.CATEGORY_COUNTS, "kind"))
        self.assertEqual([(type(label), label, count) for label, count in payload.category_counts], [
            (int, 1, 2), (str, "1", 1),
        ])
        self.assertEqual(payload.missing_count, 2)
        self.assertEqual(payload.other_count, 0)
        self.assertEqual(payload.excluded_count, 0)

    def test_category_top_n_is_count_sorted_and_rest_is_other(self):
        source = _source(columns=[_column("kind", "kind", ["a", "b", "a", "c", "d", "e"])])
        spec = PlotSpec(PlotKind.CATEGORY_COUNTS, ("kind",), top_n=2)
        payload = build_plot_payload(source, spec)
        self.assertEqual(payload.category_counts, (("a", 2), ("b", 1)))
        self.assertEqual(payload.other_count, 3)
        self.assertEqual(payload.effective_count, 6)

    def test_correlation_is_pairwise_finite_and_constants_are_undefined(self):
        source = _source(columns=[
            _column("x", "x", [1, 2, 3, float("nan")], "float64"),
            _column("y", "y", [2, 4, 8, 10], "float64"),
            _column("constant", "constant", [7, 7, 7, 7], "int64"),
        ])
        payload = build_plot_payload(source, _spec(source, PlotKind.CORRELATION, "x", "y", "constant"))
        pair = next(item for item in payload.correlation_counts if item.left_column_id == "x" and item.right_column_id == "y")
        self.assertEqual((pair.effective_count, pair.excluded_count), (3, 1))
        self.assertAlmostEqual(pair.correlation, 0.9819805060619657)
        constant_pair = next(
            item for item in payload.correlation_counts
            if item.left_column_id == "x" and item.right_column_id == "constant"
        )
        self.assertIsNone(constant_pair.correlation)
        self.assertEqual((payload.effective_count, payload.excluded_count), (3, 1))

    def test_confusion_matrix_uses_typed_stable_category_order(self):
        source = _source(
            kind="session_cache",
            partition="validation",
            columns=(),
            outputs=[
                _column("classification.y_true", "true", [1, "1", 1, None]),
                _column("classification.y_pred", "predicted", ["1", 1, 1, 0]),
            ],
        )
        payload = build_plot_payload(
            source,
            _spec(source, PlotKind.CLASSIFICATION_CONFUSION, "classification.y_true", "classification.y_pred"),
        )
        self.assertEqual([(type(item), item) for item in payload.labels], [(int, 1), (str, "1")])
        self.assertEqual(payload.matrix, ((1, 1), (1, 0)))
        self.assertEqual((payload.effective_count, payload.excluded_count), (3, 1))

    def test_regression_residual_is_true_minus_prediction_and_position_aligned(self):
        source = _source(
            kind="session_cache",
            partition="test",
            positions=(8, 2, 11, 1),
            outputs=[
                _column("regression.y_true", "target", [10, 20, float("nan"), float("inf")]),
                _column("regression.y_pred", "prediction", [8, 17, 5, 2]),
            ],
        )
        payload = build_plot_payload(
            source,
            _spec(source, PlotKind.REGRESSION_RESIDUAL, "regression.y_true", "regression.y_pred"),
        )
        self.assertEqual(payload.x_values, (10.0, 20.0))
        self.assertEqual(payload.y_values, (2.0, 3.0))
        self.assertEqual((payload.effective_count, payload.excluded_count), (2, 2))

    def test_cluster_chart_uses_existing_two_feature_columns_and_owned_labels(self):
        source = _source(
            kind="session_cache",
            partition="train",
            columns=[
                _column("feature.x", "x", [1, 2, 3], "int64"),
                _column("feature.y", "y", [5, 4, 3], "int64"),
            ],
            outputs=[_column("clustering.label", "cluster", [0, "0", 0])],
        )
        payload = build_plot_payload(
            source,
            _spec(source, PlotKind.CLUSTER_SCATTER, "feature.x", "feature.y", "clustering.label"),
        )
        self.assertEqual([(series.name, series.x_values, series.y_values) for series in payload.series], [
            ("0 [int]", (1.0, 3.0), (5.0, 3.0)),
            ("'0' [str]", (2.0,), (4.0,)),
        ])

    def test_hmm_payload_orders_groups_and_does_not_encode_state_as_classification(self):
        source = PlotSource(
            schema_version=1,
            source_kind="session_cache",
            source_id="hmm-session",
            owner_kind="session",
            owner_id="session-1",
            partition="test",
            row_positions=(10, 11, 12),
            columns=(),
            outputs=(
                _column("hmm.hidden_state", "hidden state", [0, 1, 0]),
                _column("hmm.posterior.0", "posterior 0", [0.9, 0.2, 0.8], "float64"),
                _column("hmm.posterior.1", "posterior 1", [0.1, 0.8, 0.2], "float64"),
            ),
            sequence_map=(
                PlotSequenceMap(10, group_value="A", time_value=20, order_value_ns=20),
                PlotSequenceMap(11, group_value="A", time_value=10, order_value_ns=10),
                PlotSequenceMap(12, group_value="B", time_value=5, order_value_ns=5),
            ),
            partition_count=3,
        )
        payload = build_plot_payload(source, _spec(
            source, PlotKind.HMM_STATE_POSTERIOR,
            "hmm.hidden_state", "hmm.posterior.0", "hmm.posterior.1",
        ))
        self.assertEqual(payload.x_values, (10, 20, 5))
        self.assertEqual(payload.y_values, (1, 0, 0))
        self.assertEqual(payload.labels, ("A", "A", "B"))
        self.assertTrue(all(column_id.startswith("hmm.") for column_id in [item.column_id for item in source.outputs]))
        self.assertNotIn(PlotKind.CLASSIFICATION_CONFUSION, {item.kind for item in available_plot_specs(source)})

    def test_hmm_utc_nanosecond_order_values_render_as_readable_utc_dates(self):
        timestamps_ns = (1_704_067_200_000_000_000, 1_704_067_260_000_000_000)
        source = PlotSource(
            schema_version=1,
            source_kind="session_cache",
            source_id="hmm-session",
            owner_kind="session",
            owner_id="session-utc",
            partition="validation",
            row_positions=(4, 9),
            outputs=(
                _column("hmm.hidden_state", "hidden state", [0, 1], "int64"),
                _column("hmm.posterior.0", "posterior 0", [0.8, 0.2], "float64"),
            ),
            sequence_map=tuple(
                PlotSequenceMap(position, group_value="device-a", order_value_ns=timestamp)
                for position, timestamp in zip((4, 9), timestamps_ns)
            ),
            partition_count=2,
        )
        payload = build_plot_payload(source, _spec(
            source, PlotKind.HMM_STATE_POSTERIOR,
            "hmm.hidden_state", "hmm.posterior.0",
        ))
        self.assertEqual(payload.x_values, timestamps_ns)
        self.assertEqual(payload.x_label, "时间（UTC）")
        self.assertIn("规范化显示", payload.note)

        figure = render_plot_payload(payload)
        state_axis, posterior_axis = figure.axes
        expected = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(tuple(state_axis.lines[0].get_xdata(orig=True)), expected)
        self.assertEqual(tuple(posterior_axis.lines[0].get_xdata(orig=True)), expected)
        self.assertEqual(posterior_axis.get_xlabel(), "时间（UTC）")

    def test_invalid_or_unavailable_specs_are_rejected(self):
        source = _source(columns=[_column("x", "x", [1, 2], "int64")])
        with self.assertRaises(ValueError):
            build_plot_payload(source, PlotSpec(PlotKind.SCATTER, ("x", "missing")))
        with self.assertRaises(ValueError):
            PlotSpec(PlotKind.HISTOGRAM, ("x",), bins=0)

    def test_base_import_and_payload_builder_do_not_import_matplotlib(self):
        code = (
            "import sys; from pyml_workbench.plotting import PlotColumn, PlotSource, PlotSpec, PlotKind, build_plot_payload; "
            "s=PlotSource(1,'loaded_eda','s','loaded_dataset','d','all',(0,),"
            "columns=(PlotColumn('x','x','int64',(1,)),),partition_count=1); "
            "build_plot_payload(s,PlotSpec(PlotKind.HISTOGRAM,('x',))); "
            "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_renderer_targets_explicit_figure_and_exports_png_svg(self):
        source = _source(columns=[_column("x", "x", [1, 2, 3], "int64")])
        payload = build_plot_payload(source, _spec(source, PlotKind.HISTOGRAM, "x"))
        from matplotlib.figure import Figure

        first = Figure()
        second = Figure()
        self.assertIs(render_plot_payload(payload, second), second)
        self.assertEqual(len(first.axes), 0)
        with tempfile.TemporaryDirectory() as directory:
            png = Path(directory) / "selected.png"
            svg = Path(directory) / "selected.svg"
            second.savefig(png, format="png")
            second.savefig(svg, format="svg")
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn(b"<svg", svg.read_bytes()[:1000])

    def test_renderer_smoke_covers_every_allowlisted_chart_kind(self):
        source = PlotSource(
            schema_version=1,
            source_kind="session_cache",
            source_id="render-source",
            owner_kind="session",
            owner_id="session-1",
            partition="validation",
            row_positions=(0, 1, 2),
            columns=(
                _column("x", "x", [1, 2, 3], "int64"),
                _column("y", "y", [4, 2, 5], "int64"),
                _column("category", "category", ["a", "b", "a"]),
            ),
            outputs=(
                _column("classification.y_true", "true label", ["a", "b", "a"]),
                _column("classification.y_pred", "predicted label", ["a", "a", "a"]),
                _column("regression.y_true", "true value", [1.0, 2.0, 3.0], "float64"),
                _column("regression.y_pred", "predicted value", [1.2, 1.8, 2.5], "float64"),
                _column("clustering.label", "cluster", [0, 1, 0], "int64"),
                _column("hmm.hidden_state", "hidden state", [0, 1, 0], "int64"),
                _column("hmm.posterior.0", "posterior 0", [0.8, 0.2, 0.7], "float64"),
                _column("hmm.posterior.1", "posterior 1", [0.2, 0.8, 0.3], "float64"),
            ),
            sequence_map=tuple(
                PlotSequenceMap(index, group_value="g", time_value=index, order_value_ns=index)
                for index in range(3)
            ),
            partition_count=3,
        )
        requested = (
            (PlotKind.HISTOGRAM, ("x",)),
            (PlotKind.CATEGORY_COUNTS, ("category",)),
            (PlotKind.SCATTER, ("x", "y")),
            (PlotKind.CORRELATION, ("x", "y")),
            (PlotKind.CLASSIFICATION_CONFUSION, ("classification.y_true", "classification.y_pred")),
            (PlotKind.REGRESSION_ACTUAL_PREDICTED, ("regression.y_true", "regression.y_pred")),
            (PlotKind.REGRESSION_RESIDUAL, ("regression.y_true", "regression.y_pred")),
            (PlotKind.CLUSTER_SCATTER, ("x", "y", "clustering.label")),
            (PlotKind.HMM_STATE_POSTERIOR, ("hmm.hidden_state", "hmm.posterior.0", "hmm.posterior.1")),
        )
        for kind, column_ids in requested:
            with self.subTest(kind=kind.value):
                payload = build_plot_payload(source, _spec(source, kind, *column_ids))
                figure = render_plot_payload(payload)
                self.assertGreaterEqual(len(figure.axes), 1)


if __name__ == "__main__":
    unittest.main()
