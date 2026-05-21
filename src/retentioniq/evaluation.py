from __future__ import annotations

import numpy as np
import pandas as pd

from retentioniq.metrics import (
    average_precision,
    brier_score,
    calibration_table,
    capture_at_k,
    expected_calibration_error,
    lift_at_k,
    log_loss,
    optimize_retention_threshold,
    roc_auc,
)


def heuristic_risk_score(frame: pd.DataFrame) -> np.ndarray:
    """A transparent baseline that uses hand-built business rules."""

    risk = (
        -1.6 * frame["active_rate_recent_3m"]
        - 0.8 * frame["feature_velocity_change_3m"]
        - 0.12 * frame["nps_recent_3m"]
        + 1.8 * frame["invoice_overdue_recent_3m"]
        + 0.55 * frame["support_ticket_intensity"]
        + 0.48 * frame["near_renewal"]
        + 0.32 * frame["segment_label"].eq("SMB").astype(float)
    )
    risk = np.asarray(risk, dtype=float)
    centered = (risk - risk.mean()) / (risk.std() if risk.std() else 1.0)
    return 1.0 / (1.0 + np.exp(-centered))


def evaluate_churn_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_value: np.ndarray,
    threshold: float | None = None,
) -> dict[str, object]:
    threshold_result = optimize_retention_threshold(y_true, scores, customer_value)
    chosen_threshold = threshold if threshold is not None else float(threshold_result["threshold"])
    return {
        "roc_auc": roc_auc(y_true, scores),
        "average_precision": average_precision(y_true, scores),
        "brier_score": brier_score(y_true, scores),
        "log_loss": log_loss(y_true, scores),
        "expected_calibration_error": expected_calibration_error(y_true, scores),
        "lift_at_10_pct": lift_at_k(y_true, scores, 0.10),
        "capture_at_10_pct": capture_at_k(y_true, scores, 0.10),
        "lift_at_20_pct": lift_at_k(y_true, scores, 0.20),
        "capture_at_20_pct": capture_at_k(y_true, scores, 0.20),
        "optimized_threshold": threshold_result,
        "selected_threshold": chosen_threshold,
        "calibration_bins": calibration_table(y_true, scores, bins=10),
    }


def permutation_importance(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    baseline = roc_auc(y, model.predict_proba(x))
    rows: list[dict[str, float | str]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(n_repeats):
            permuted = x.copy()
            permuted[:, feature_idx] = rng.permutation(permuted[:, feature_idx])
            permuted_auc = roc_auc(y, model.predict_proba(permuted))
            drops.append(float(baseline - permuted_auc))
        rows.append(
            {
                "feature": feature_name,
                "mean_auc_drop": float(np.mean(drops)),
                "std_auc_drop": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)


def segment_performance(scored: pd.DataFrame, segment_col: str = "segment_label") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, group in scored.groupby(segment_col):
        if len(group) < 20:
            continue
        rows.append(
            {
                "segment": segment,
                "customers": int(len(group)),
                "actual_churn_rate": float(group["churn_next_90d"].mean()),
                "avg_predicted_risk": float(group["predicted_churn_risk"].mean()),
                "avg_predicted_uplift": float(group["predicted_offer_uplift"].mean()),
                "value_at_risk": float(group["expected_value_at_risk"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("value_at_risk", ascending=False).reset_index(drop=True)
