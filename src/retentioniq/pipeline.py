from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retentioniq.config import ARTIFACT_DIR, PROCESSED_DIR, RAW_DIR, REPORT_DIR, ensure_project_dirs
from retentioniq.data import SyntheticDataConfig, generate_raw_data
from retentioniq.evaluation import (
    evaluate_churn_scores,
    heuristic_risk_score,
    permutation_importance,
    segment_performance,
)
from retentioniq.features import build_feature_table, get_feature_columns, temporal_train_test_split
from retentioniq.metrics import uplift_policy_value
from retentioniq.models import fit_probability_pipeline, fit_t_learner_uplift
from retentioniq.monitoring import population_stability_report
from retentioniq.reporting import render_dashboard, render_model_card, write_json


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _to_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return frame[feature_columns].to_numpy(dtype=float)


def _recommended_actions(scores: np.ndarray, uplift: np.ndarray, threshold: float) -> np.ndarray:
    actions = np.full(len(scores), "Monitor", dtype=object)
    actions[(scores >= threshold) & (uplift > 0.025)] = "Offer + CSM outreach"
    actions[(scores >= threshold) & (uplift <= 0.025)] = "CSM outreach"
    actions[(scores < threshold) & (scores >= np.quantile(scores, 0.75))] = "Health check"
    return actions


def run_pipeline(customers: int, seed: int) -> dict[str, Path]:
    ensure_project_dirs()

    generate_raw_data(
        SyntheticDataConfig(n_customers=customers, random_seed=seed),
        output_dir=RAW_DIR,
    )
    feature_frame = build_feature_table(RAW_DIR, PROCESSED_DIR / "customer_features.csv")
    feature_columns = get_feature_columns(feature_frame)
    train_df, holdout_df = temporal_train_test_split(feature_frame, test_fraction=0.25)

    train_cut = int(len(train_df) * 0.82)
    model_train_df = train_df.iloc[:train_cut].copy()
    calibration_df = train_df.iloc[train_cut:].copy()

    x_train = _to_matrix(model_train_df, feature_columns)
    y_train = model_train_df["churn_next_90d"].to_numpy(dtype=int)
    x_calibration = _to_matrix(calibration_df, feature_columns)
    y_calibration = calibration_df["churn_next_90d"].to_numpy(dtype=int)
    x_holdout = _to_matrix(holdout_df, feature_columns)
    y_holdout = holdout_df["churn_next_90d"].to_numpy(dtype=int)
    customer_value = (holdout_df["mrr"].to_numpy(dtype=float) * 3.0)

    churn_model = fit_probability_pipeline(
        x_train,
        y_train,
        feature_columns,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
        random_seed=seed,
    )
    churn_scores = churn_model.predict_proba(x_holdout)
    baseline_scores = heuristic_risk_score(holdout_df)

    train_x_full = _to_matrix(train_df, feature_columns)
    uplift_model = fit_t_learner_uplift(
        train_x_full,
        train_df["churn_next_90d"].to_numpy(dtype=int),
        train_df["offer_sent"].to_numpy(dtype=int),
        feature_columns,
        random_seed=seed,
    )
    predicted_uplift = uplift_model.predict_uplift(x_holdout)
    assumed_offer_cost = holdout_df["mrr"].to_numpy(dtype=float) * 0.15 * 3.0

    churn_metrics = evaluate_churn_scores(y_holdout, churn_scores, customer_value)
    baseline_metrics = evaluate_churn_scores(y_holdout, baseline_scores, customer_value)
    uplift_metrics = uplift_policy_value(
        y_holdout,
        predicted_uplift,
        customer_value,
        assumed_offer_cost,
        top_fraction=0.20,
    )
    metrics = {
        "dataset": {
            "customers": int(len(feature_frame)),
            "features": int(len(feature_columns)),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_churn_rate": float(y_holdout.mean()),
        },
        "churn_model": churn_metrics,
        "heuristic_baseline": baseline_metrics,
        "uplift_policy": uplift_metrics,
    }

    importance = permutation_importance(churn_model, x_holdout, y_holdout, feature_columns, n_repeats=3, random_seed=seed)
    drift_report = population_stability_report(train_df, holdout_df, feature_columns)

    scored = holdout_df[
        [
            "customer_id",
            "signup_date",
            "segment_label",
            "plan_label",
            "region_label",
            "mrr",
            "churn_next_90d",
            "offer_sent",
        ]
    ].copy()
    scored["predicted_churn_risk"] = np.round(churn_scores, 5)
    scored["predicted_offer_uplift"] = np.round(predicted_uplift, 5)
    scored["expected_value_at_risk"] = np.round(churn_scores * customer_value, 2)
    scored["retention_priority_score"] = np.round(
        scored["expected_value_at_risk"] * (1 + np.clip(predicted_uplift, -0.25, 0.50)),
        2,
    )
    threshold = float(churn_metrics["optimized_threshold"]["threshold"])
    scored["recommended_action"] = _recommended_actions(churn_scores, predicted_uplift, threshold)
    scored = scored.sort_values("retention_priority_score", ascending=False)

    segment_perf = segment_performance(scored)

    churn_model.save(ARTIFACT_DIR / "churn_model.npz")
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    drift_report.to_csv(ARTIFACT_DIR / "drift_report.csv", index=False)
    segment_perf.to_csv(ARTIFACT_DIR / "segment_performance.csv", index=False)
    scored.to_csv(ARTIFACT_DIR / "scored_holdout_customers.csv", index=False)
    write_json(_json_ready(metrics), ARTIFACT_DIR / "metrics.json")
    render_model_card(_json_ready(metrics), importance, ARTIFACT_DIR / "model_card.md")
    render_dashboard(
        _json_ready(metrics),
        scored,
        importance,
        segment_perf,
        drift_report,
        REPORT_DIR / "dashboard.html",
    )

    return {
        "features": PROCESSED_DIR / "customer_features.csv",
        "metrics": ARTIFACT_DIR / "metrics.json",
        "model": ARTIFACT_DIR / "churn_model.npz",
        "scored_customers": ARTIFACT_DIR / "scored_holdout_customers.csv",
        "dashboard": REPORT_DIR / "dashboard.html",
        "model_card": ARTIFACT_DIR / "model_card.md",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RetentionIQ end-to-end data science pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate data, train models, evaluate, and render reports")
    run_parser.add_argument("--customers", type=int, default=8_000, help="Number of synthetic customers")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)
    if args.command == "run":
        outputs = run_pipeline(customers=args.customers, seed=args.seed)
        print("RetentionIQ pipeline completed.")
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
