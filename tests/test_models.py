import unittest

import numpy as np

from retentioniq.metrics import roc_auc
from retentioniq.models import fit_probability_pipeline


class ModelTests(unittest.TestCase):
    def test_probability_pipeline_learns_linear_signal(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(500, 3))
        logits = 1.8 * x[:, 0] - 1.2 * x[:, 1] + 0.3
        y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
        model = fit_probability_pipeline(x[:400], y[:400], ["a", "b", "c"], x[400:], y[400:])
        scores = model.predict_proba(x[400:])
        self.assertGreater(roc_auc(y[400:], scores), 0.75)
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
