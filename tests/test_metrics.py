import unittest

import numpy as np

from retentioniq.metrics import average_precision, lift_at_k, optimize_retention_threshold, roc_auc


class MetricTests(unittest.TestCase):
    def test_auc_is_perfect_for_perfect_ranking(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(y, scores), 1.0)

    def test_average_precision_is_high_for_good_ranking(self):
        y = np.array([1, 0, 1, 0, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        self.assertGreater(average_precision(y, scores), 0.75)

    def test_lift_at_k_prioritizes_positive_cases(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.95, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.assertGreater(lift_at_k(y, scores, 0.3), 2.0)

    def test_profit_threshold_returns_policy_fields(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.7, 0.6, 0.1])
        value = np.array([1000, 800, 400, 300])
        result = optimize_retention_threshold(y, scores, value)
        self.assertIn("threshold", result)
        self.assertIn("profit", result)
        self.assertGreaterEqual(result["selected_customers"], 1)


if __name__ == "__main__":
    unittest.main()
