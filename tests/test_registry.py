import unittest
from unittest.mock import patch

from pyml_workbench import (
    build_estimator,
    get_model,
    list_models,
    model_capabilities,
    parameter_schema,
)
from pyml_workbench.sequence_models import (
    EXTENDED_MODEL_IDS,
    OptionalModelDependencyError,
    build_hmm_estimator,
)


class RegistryTests(unittest.TestCase):
    def test_catalog_counts_and_every_traditional_estimator_constructs(self):
        models = list_models()
        self.assertEqual(len(models), 78)
        expected = {
            "classification": 25,
            "regression": 26,
            "clustering": 10,
            "dimensionality reduction": 10,
            "anomaly detection": 4,
            "sequence_modeling": 3,
        }
        for task, count in expected.items():
            self.assertEqual(sum(item["task"] == task for item in models), count)
        self.assertEqual(len({item["id"] for item in models}), 78)
        for item in models:
            if item["id"] not in EXTENDED_MODEL_IDS:
                estimator = build_estimator(item["id"])
                self.assertEqual(estimator.__class__.__name__, item["class_name"])
            self.assertIsInstance(parameter_schema(item["id"]), list)

    def test_extended_models_are_in_the_default_runnable_catalog(self):
        all_items = list_models(include_deferred=True)
        self.assertEqual(len(all_items), 78)
        self.assertTrue(all(item["implementation_status"] == "available" for item in all_items))
        self.assertEqual({item["id"] for item in all_items if item["id"] in EXTENDED_MODEL_IDS}, set(EXTENDED_MODEL_IDS))
        self.assertEqual(len(list_models()), 78)
        for model_id in EXTENDED_MODEL_IDS:
            self.assertEqual(get_model(model_id)["implementation_status"], "available")

    def test_optional_hmm_extra_is_lazy_and_does_not_hide_catalog_entry(self):
        with patch("pyml_workbench.sequence_models.importlib.import_module", side_effect=ModuleNotFoundError("hmmlearn")):
            self.assertEqual(get_model("H01")["implementation_status"], "available")
            self.assertIn("H01", {item["id"] for item in list_models()})
            with self.assertRaisesRegex(OptionalModelDependencyError, "sequence.*extra"):
                build_hmm_estimator("H01", {"n_components": 2})

    def test_capability_boundaries_are_real(self):
        self.assertFalse(model_capabilities("C02")["predict_proba"])
        self.assertFalse(model_capabilities("C07")["predict_proba"])
        self.assertTrue(model_capabilities("C07", build_estimator("C07", {"probability": True}))["predict_proba"])
        self.assertFalse(model_capabilities("K07")["predict"])
        self.assertFalse(model_capabilities("D10")["transform"])
        self.assertFalse(model_capabilities("A03")["predict"])
        self.assertTrue(model_capabilities("A03", build_estimator("A03", {"novelty": True}))["predict"])

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported parameters"):
            build_estimator("C01", {"not_a_real_parameter": 1})


if __name__ == "__main__":
    unittest.main()
