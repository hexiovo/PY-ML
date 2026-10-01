"""Comparison-group rank semantics for persisted batch winners."""
import unittest

from pyml_workbench.batch import _finite_objective_score, _rank_validation_group_entries


class BatchLeaderboardRankTests(unittest.TestCase):
    def test_competition_ranks_respect_direction_and_exclude_unrankable_scores(self):
        entries = [
            {"job_id": "max-third", "comparison_group_key": "max", "direction": "max", "score": 0.7},
            {"job_id": "max-tie-b", "comparison_group_key": "max", "direction": "max", "score": 0.9},
            {"job_id": "max-tie-a", "comparison_group_key": "max", "direction": "max", "score": 0.9},
            {"job_id": "max-no-score", "comparison_group_key": "max", "direction": "max", "score": None},
            {"job_id": "rmse-low", "comparison_group_key": "rmse", "direction": "min", "score": 0.2},
            {"job_id": "rmse-high", "comparison_group_key": "rmse", "direction": "min", "score": 0.8},
            {
                "job_id": "train-only",
                "comparison_group_key": "exploration",
                "direction": "max",
                "score_split": "train_exploratory",
                "score": 1.0,
            },
            {"job_id": "non-finite", "comparison_group_key": "max", "direction": "max", "score": float("nan")},
        ]

        self.assertEqual(
            _rank_validation_group_entries(entries),
            {
                "max-third": 3,
                "max-tie-b": 1,
                "max-tie-a": 1,
                "max-no-score": None,
                "rmse-low": 1,
                "rmse-high": 2,
                "train-only": None,
                "non-finite": None,
            },
        )
        self.assertIsNone(_finite_objective_score(float("inf")))
        self.assertIsNone(_finite_objective_score(True))


if __name__ == "__main__":
    unittest.main()
