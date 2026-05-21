from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retentioniq.config import OBSERVATION_END, PROCESSED_DIR, RAW_DIR, ensure_project_dirs


NON_FEATURE_COLUMNS = {
    "customer_id",
    "signup_date",
    "churn_next_90d",
    "offer_sent",
    "offer_discount_pct",
    "offer_channel",
    "offer_cost",
    "days_until_churn",
    "realized_90d_revenue",
    "true_churn_probability_no_offer",
    "true_churn_probability_observed",
}


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def load_raw_tables(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(raw_dir / "customers.csv", parse_dates=["signup_date"])
    usage = pd.read_csv(raw_dir / "monthly_usage.csv")
    campaign = pd.read_csv(raw_dir / "retention_campaign.csv")
    return customers, usage, campaign


def _wide_slope(usage: pd.DataFrame, value_col: str) -> pd.Series:
    wide = usage.pivot(index="customer_id", columns="month_index", values=value_col).sort_index(axis=1)
    y = wide.to_numpy(dtype=float)
    x = np.arange(y.shape[1], dtype=float)
    x_centered = x - x.mean()
    y_centered = y - np.nanmean(y, axis=1, keepdims=True)
    denom = float((x_centered**2).sum())
    slopes = np.nan_to_num((y_centered @ x_centered) / denom, nan=0.0)
    return pd.Series(slopes, index=wide.index, name=f"{value_col}_slope")


def build_feature_table(
    raw_dir: Path = RAW_DIR,
    output_path: Path | None = PROCESSED_DIR / "customer_features.csv",
    observation_end: str = OBSERVATION_END,
) -> pd.DataFrame:
    """Create customer-level, leakage-safe features from raw operational tables."""

    ensure_project_dirs()
    customers, usage, campaign = load_raw_tables(raw_dir)
    usage = usage.merge(customers[["customer_id", "seats"]], on="customer_id", how="left")

    usage["active_rate"] = safe_divide(usage["active_users"], usage["seats"])
    usage["logins_per_active_user"] = safe_divide(usage["logins"], usage["active_users"])
    usage["feature_events"] = usage["collaboration_events"] + usage["automation_runs"] + usage["api_calls"]
    usage["feature_events_per_seat"] = safe_divide(usage["feature_events"], usage["seats"])
    usage["support_tickets_per_seat"] = safe_divide(usage["support_tickets"], usage["seats"])
    usage["severe_ticket_rate"] = safe_divide(usage["severe_tickets"], usage["support_tickets"])

    max_month = int(usage["month_index"].max())
    recent_mask = usage["month_index"] >= max_month - 2
    earlier_mask = usage["month_index"] < max_month - 2

    last = usage[usage["month_index"] == max_month].set_index("customer_id")
    recent = usage[recent_mask].groupby("customer_id").mean(numeric_only=True)
    earlier = usage[earlier_mask].groupby("customer_id").mean(numeric_only=True)
    six_month = usage.groupby("customer_id").agg(
        support_tickets_6m=("support_tickets", "sum"),
        severe_tickets_6m=("severe_tickets", "sum"),
        overdue_months_6m=("invoice_overdue", "sum"),
        nps_min_6m=("nps", "min"),
        nps_max_6m=("nps", "max"),
        feature_events_6m=("feature_events", "sum"),
        active_users_mean_6m=("active_users", "mean"),
    )

    feature_frame = customers.set_index("customer_id").join(campaign.set_index("customer_id"), how="inner")
    feature_frame["tenure_months"] = (
        pd.Timestamp(observation_end) - pd.to_datetime(feature_frame["signup_date"])
    ).dt.days / 30.4375
    feature_frame["is_annual_contract"] = (feature_frame["contract_months"] >= 12).astype(int)
    feature_frame["log_mrr"] = np.log1p(feature_frame["mrr"])
    feature_frame["log_seats"] = np.log1p(feature_frame["seats"])
    feature_frame["mrr_per_seat"] = safe_divide(feature_frame["mrr"], feature_frame["seats"])

    current_month = pd.Timestamp(observation_end).month
    feature_frame["months_to_renewal"] = ((feature_frame["renewal_month"] - current_month) % 12).astype(float)
    feature_frame["near_renewal"] = (feature_frame["months_to_renewal"] <= 2).astype(int)

    for source, suffix in [(last, "last"), (recent, "recent_3m"), (earlier, "prior_3m")]:
        feature_frame[f"active_rate_{suffix}"] = source["active_rate"]
        feature_frame[f"logins_per_active_user_{suffix}"] = source["logins_per_active_user"]
        feature_frame[f"feature_events_per_seat_{suffix}"] = source["feature_events_per_seat"]
        feature_frame[f"support_tickets_per_seat_{suffix}"] = source["support_tickets_per_seat"]
        feature_frame[f"nps_{suffix}"] = source["nps"]
        feature_frame[f"invoice_overdue_{suffix}"] = source["invoice_overdue"]
        feature_frame[f"automation_runs_{suffix}"] = source["automation_runs"]
        feature_frame[f"api_calls_{suffix}"] = source["api_calls"]

    feature_frame = feature_frame.join(six_month, how="left")
    feature_frame["active_rate_change_3m"] = feature_frame["active_rate_recent_3m"] - feature_frame["active_rate_prior_3m"]
    feature_frame["nps_change_3m"] = feature_frame["nps_recent_3m"] - feature_frame["nps_prior_3m"]
    feature_frame["feature_velocity_change_3m"] = (
        feature_frame["feature_events_per_seat_recent_3m"] - feature_frame["feature_events_per_seat_prior_3m"]
    )
    feature_frame["support_ticket_intensity"] = safe_divide(feature_frame["support_tickets_6m"], feature_frame["seats"])
    feature_frame["severe_ticket_share_6m"] = safe_divide(feature_frame["severe_tickets_6m"], feature_frame["support_tickets_6m"])
    feature_frame["feature_depth_score"] = (
        np.log1p(feature_frame["automation_runs_recent_3m"])
        + np.log1p(feature_frame["api_calls_recent_3m"])
        + 2.5 * feature_frame["feature_events_per_seat_recent_3m"]
    )

    for value_col in ["active_rate", "logins_per_active_user", "feature_events_per_seat", "support_tickets_per_seat", "nps"]:
        feature_frame = feature_frame.join(_wide_slope(usage, value_col), how="left")

    feature_frame["segment_label"] = feature_frame["segment"]
    feature_frame["plan_label"] = feature_frame["plan"]
    feature_frame["region_label"] = feature_frame["region"]

    categorical_cols = [
        "segment",
        "plan",
        "industry",
        "region",
        "acquisition_channel",
        "customer_health_owner",
    ]
    encoded = pd.get_dummies(feature_frame[categorical_cols], prefix=categorical_cols, dtype=int)
    feature_frame = pd.concat([feature_frame.drop(columns=categorical_cols), encoded], axis=1)

    feature_frame = feature_frame.reset_index()
    numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns
    feature_frame[numeric_cols] = feature_frame[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_frame.to_csv(output_path, index=False)

    return feature_frame


def get_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in NON_FEATURE_COLUMNS]


def temporal_train_test_split(
    feature_frame: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = feature_frame.sort_values("signup_date").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
