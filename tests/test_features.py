import tempfile
import unittest
from pathlib import Path

from retentioniq.data import SyntheticDataConfig, generate_raw_data
from retentioniq.features import build_feature_table, get_feature_columns, temporal_train_test_split


class FeatureTests(unittest.TestCase):
    def test_feature_table_has_labels_and_numeric_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            generate_raw_data(SyntheticDataConfig(n_customers=250, random_seed=5), raw_dir)
            features = build_feature_table(raw_dir, output_path=None)
            feature_columns = get_feature_columns(features)
            self.assertIn("churn_next_90d", features.columns)
            self.assertIn("active_rate_recent_3m", feature_columns)
            self.assertGreater(len(feature_columns), 20)
            self.assertEqual(features[feature_columns].isna().sum().sum(), 0)

    def test_temporal_split_partitions_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            generate_raw_data(SyntheticDataConfig(n_customers=200, random_seed=6), raw_dir)
            features = build_feature_table(raw_dir, output_path=None)
            train, holdout = temporal_train_test_split(features, test_fraction=0.25)
            self.assertEqual(len(train) + len(holdout), len(features))
            self.assertLessEqual(train["signup_date"].max(), holdout["signup_date"].max())


if __name__ == "__main__":
    unittest.main()
