from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if y.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    return y, score


def roc_auc(y_true, y_score) -> float:
    y, score = _as_arrays(y_true, y_score)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos_rank_sum = ranks[positives].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, y_score) -> float:
    y, score = _as_arrays(y_true, y_score)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    recall = tp / positives
    recall_step = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_step))


def brier_score(y_true, y_score) -> float:
    y, score = _as_arrays(y_true, y_score)
    return float(np.mean((score - y) ** 2))


def log_loss(y_true, y_score, eps: float = 1e-15) -> float:
    y, score = _as_arrays(y_true, y_score)
    score = np.clip(score, eps, 1 - eps)
    return float(-np.mean(y * np.log(score) + (1 - y) * np.log(1 - score)))


def calibration_table(y_true, y_score, bins: int = 10) -> list[dict[str, float]]:
    y, score = _as_arrays(y_true, y_score)
    frame = pd.DataFrame({"y": y, "score": score})
    frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    rows: list[dict[str, float]] = []
    for _, group in grouped:
        rows.append(
            {
                "count": int(len(group)),
                "avg_prediction": float(group["score"].mean()),
                "observed_rate": float(group["y"].mean()),
                "min_prediction": float(group["score"].min()),
                "max_prediction": float(group["score"].max()),
            }
        )
    return rows


def expected_calibration_error(y_true, y_score, bins: int = 10) -> float:
    table = calibration_table(y_true, y_score, bins=bins)
    total = sum(row["count"] for row in table)
    if total == 0:
        return float("nan")
    return float(
        sum(row["count"] * abs(row["observed_rate"] - row["avg_prediction"]) for row in table) / total
    )


def lift_at_k(y_true, y_score, k: float = 0.10) -> float:
    y, score = _as_arrays(y_true, y_score)
    selected_count = max(1, int(np.ceil(len(y) * k)))
    selected = np.argsort(-score)[:selected_count]
    base_rate = y.mean()
    if base_rate == 0:
        return float("nan")
    return float(y[selected].mean() / base_rate)


def capture_at_k(y_true, y_score, k: float = 0.10) -> float:
    y, score = _as_arrays(y_true, y_score)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    selected_count = max(1, int(np.ceil(len(y) * k)))
    selected = np.argsort(-score)[:selected_count]
    return float(y[selected].sum() / positives)


def confusion_at_threshold(y_true, y_score, threshold: float) -> dict[str, float]:
    y, score = _as_arrays(y_true, y_score)
    predicted = score >= threshold
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
    }


def optimize_retention_threshold(
    y_true,
    y_score,
    customer_value,
    save_rate: float = 0.35,
    contact_cost_rate: float = 0.04,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Find a threshold that maximizes a simple retention-campaign profit model."""

    y, score = _as_arrays(y_true, y_score)
    value = np.asarray(customer_value, dtype=float)
    if thresholds is None:
        thresholds = np.quantile(score, np.linspace(0.05, 0.95, 91))
        thresholds = np.unique(np.round(thresholds, 6))

    best: dict[str, float] | None = {
        "threshold": 1.0,
        "selected_customers": 0,
        "selected_rate": 0.0,
        "saved_revenue": 0.0,
        "contact_cost": 0.0,
        "profit": 0.0,
    }
    for threshold in thresholds:
        selected = score >= threshold
        contact_cost = (selected * value * contact_cost_rate).sum()
        saved_revenue = (selected * (y == 1) * value * save_rate).sum()
        profit = float(saved_revenue - contact_cost)
        row = {
            "threshold": float(threshold),
            "selected_customers": int(selected.sum()),
            "selected_rate": float(selected.mean()),
            "saved_revenue": float(saved_revenue),
            "contact_cost": float(contact_cost),
            "profit": profit,
        }
        if best is None or profit > best["profit"]:
            best = row
    return best or {}


def uplift_policy_value(
    observed_churn,
    predicted_uplift,
    customer_value,
    offer_cost,
    top_fraction: float = 0.20,
) -> dict[str, float]:
    """Estimate expected value from targeting customers with highest predicted uplift."""

    churn = np.asarray(observed_churn, dtype=int)
    uplift = np.asarray(predicted_uplift, dtype=float)
    value = np.asarray(customer_value, dtype=float)
    cost = np.asarray(offer_cost, dtype=float)
    selected_count = max(1, int(np.ceil(len(churn) * top_fraction)))
    selected_idx = np.argsort(-uplift)[:selected_count]
    expected_saved = float((np.clip(uplift[selected_idx], 0, 1) * value[selected_idx]).sum())
    expected_cost = float(cost[selected_idx].sum())
    return {
        "top_fraction": float(top_fraction),
        "selected_customers": int(selected_count),
        "mean_predicted_uplift": float(uplift[selected_idx].mean()),
        "observed_churn_rate_in_policy": float(churn[selected_idx].mean()),
        "expected_saved_revenue": expected_saved,
        "expected_offer_cost": expected_cost,
        "expected_net_value": float(expected_saved - expected_cost),
    }
