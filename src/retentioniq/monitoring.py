from __future__ import annotations

import numpy as np
import pandas as pd


def _psi_for_feature(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        quantiles = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()) + 1e-6, bins + 1)
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    expected_counts = np.histogram(expected, bins=quantiles)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=quantiles)[0] / len(actual)
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def population_stability_report(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        psi = _psi_for_feature(train_frame[feature].to_numpy(), holdout_frame[feature].to_numpy(), bins=bins)
        if psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "watch"
        else:
            status = "shifted"
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": status,
                "train_mean": float(train_frame[feature].mean()),
                "holdout_mean": float(holdout_frame[feature].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
