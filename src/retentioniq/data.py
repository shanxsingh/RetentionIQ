from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from retentioniq.config import OBSERVATION_END, RAW_DIR, ensure_project_dirs


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Controls the generated business scenario."""

    n_customers: int = 8_000
    months_observed: int = 6
    random_seed: int = 42
    observation_end: str = OBSERVATION_END


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def _choice(rng: np.random.Generator, values: list[str], probs: list[float], size: int) -> np.ndarray:
    return rng.choice(np.array(values, dtype=object), size=size, p=np.array(probs))


def generate_raw_data(
    config: SyntheticDataConfig | None = None,
    output_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    """Generate synthetic customer, usage, and retention campaign tables.

    The data is intentionally synthetic so the project can be shared publicly.
    Relationships among engagement, support pain, contract pressure, and churn
    are modeled so evaluation behaves like a real retention workflow.
    """

    ensure_project_dirs()
    config = config or SyntheticDataConfig()
    rng = np.random.default_rng(config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = config.n_customers
    customer_ids = np.array([f"CUST-{i:06d}" for i in range(1, n + 1)])

    segments = _choice(rng, ["SMB", "Mid-Market", "Enterprise"], [0.58, 0.30, 0.12], n)
    plans = np.empty(n, dtype=object)
    for segment in ["SMB", "Mid-Market", "Enterprise"]:
        mask = segments == segment
        if segment == "SMB":
            plans[mask] = _choice(rng, ["Starter", "Growth", "Scale"], [0.56, 0.36, 0.08], mask.sum())
        elif segment == "Mid-Market":
            plans[mask] = _choice(rng, ["Starter", "Growth", "Scale"], [0.12, 0.62, 0.26], mask.sum())
        else:
            plans[mask] = _choice(rng, ["Growth", "Scale"], [0.26, 0.74], mask.sum())

    industries = _choice(
        rng,
        ["FinTech", "Healthcare", "Retail", "SaaS", "Manufacturing", "Education"],
        [0.16, 0.15, 0.22, 0.22, 0.14, 0.11],
        n,
    )
    regions = _choice(rng, ["North America", "EMEA", "APAC", "LATAM"], [0.52, 0.24, 0.16, 0.08], n)
    channels = _choice(rng, ["Inbound", "Outbound", "Partner", "Marketplace"], [0.44, 0.24, 0.20, 0.12], n)

    segment_seat_mean = {"SMB": 18, "Mid-Market": 84, "Enterprise": 360}
    seats = np.array([rng.poisson(segment_seat_mean[s]) + 3 for s in segments]).astype(int)
    seats = np.clip(seats, 5, 900)

    plan_price = {"Starter": 32, "Growth": 58, "Scale": 91}
    plan_multiplier = np.array([plan_price[p] for p in plans])
    discount = np.clip(rng.beta(2, 10, n) + (segments == "Enterprise") * rng.uniform(0.03, 0.14, n), 0, 0.38)
    mrr = seats * plan_multiplier * (1 - discount) + rng.normal(0, 80, n)
    mrr = np.round(np.clip(mrr, 120, None), 2)

    end = pd.Timestamp(config.observation_end)
    signup_offsets = rng.integers(150, 1_650, n)
    signup_dates = end - pd.to_timedelta(signup_offsets, unit="D")
    contract_months = np.where(segments == "Enterprise", rng.choice([12, 24, 36], n, p=[0.34, 0.46, 0.20]), rng.choice([1, 12], n, p=[0.62, 0.38]))
    renewal_month = rng.integers(1, 13, n)
    customer_health_owner = _choice(rng, ["pooled", "named_csm", "strategic_csm"], [0.55, 0.34, 0.11], n)

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates,
            "segment": segments,
            "plan": plans,
            "industry": industries,
            "region": regions,
            "acquisition_channel": channels,
            "seats": seats,
            "mrr": mrr,
            "contract_months": contract_months,
            "renewal_month": renewal_month,
            "customer_health_owner": customer_health_owner,
        }
    )

    latent_fit = rng.normal(0, 0.9, n)
    price_sensitivity = sigmoid(
        -0.35
        + (segments == "SMB") * 0.75
        + (plans == "Starter") * 0.35
        + rng.normal(0, 0.55, n)
    )
    base_adoption = sigmoid(
        0.20
        + latent_fit
        + (plans == "Scale") * 0.45
        + (customer_health_owner == "strategic_csm") * 0.32
        - (channels == "Outbound") * 0.22
        + rng.normal(0, 0.55, n)
    )
    adoption_trend = rng.normal(0, 0.10, n) + (latent_fit < -0.3) * rng.normal(-0.08, 0.07, n)
    support_pain = sigmoid(
        -1.05
        - latent_fit * 0.62
        + (industries == "Healthcare") * 0.25
        + (plans == "Starter") * 0.22
        + rng.normal(0, 0.42, n)
    )

    months = pd.period_range(end=end.to_period("M"), periods=config.months_observed, freq="M")
    usage_frames: list[pd.DataFrame] = []
    for month_idx, month in enumerate(months):
        recency = month_idx - (config.months_observed - 1)
        seasonal = 0.05 * np.sin((month.month / 12) * 2 * np.pi)
        adoption = np.clip(base_adoption + adoption_trend * month_idx + seasonal + rng.normal(0, 0.05, n), 0.02, 0.98)
        active_users = np.maximum(1, rng.binomial(seats, adoption))
        logins = rng.poisson(active_users * np.clip(6.5 + 2.7 * adoption + rng.normal(0, 0.8, n), 0.4, None))
        collaboration_events = rng.poisson(active_users * np.clip(9.0 * adoption + 1.8, 0.2, None))
        automation_runs = rng.poisson(active_users * np.clip(3.5 * adoption + (plans == "Scale") * 2.0, 0.1, None))
        api_calls = rng.poisson(active_users * np.clip(4.0 * adoption + (industries == "SaaS") * 5.5, 0.0, None))
        support_tickets = rng.poisson(np.clip(0.22 + support_pain * 2.4 + (active_users / np.maximum(seats, 1)) * 0.2, 0.05, None))
        severe_tickets = rng.binomial(support_tickets, np.clip(0.07 + support_pain * 0.18, 0.01, 0.45))
        overdue_probability = sigmoid(-3.0 + price_sensitivity * 1.7 + support_pain * 0.9 + rng.normal(0, 0.35, n))
        invoice_overdue = rng.binomial(1, np.clip(overdue_probability, 0.01, 0.55))
        nps = np.round(np.clip(4.7 + 5.0 * adoption - 2.4 * support_pain + rng.normal(0, 1.2, n), 0, 10), 1)

        usage_frames.append(
            pd.DataFrame(
                {
                    "customer_id": customer_ids,
                    "usage_month": str(month),
                    "month_index": month_idx,
                    "active_users": active_users,
                    "logins": logins,
                    "collaboration_events": collaboration_events,
                    "automation_runs": automation_runs,
                    "api_calls": api_calls,
                    "support_tickets": support_tickets,
                    "severe_tickets": severe_tickets,
                    "invoice_overdue": invoice_overdue,
                    "nps": nps,
                    "recency_index": recency,
                }
            )
        )

    usage = pd.concat(usage_frames, ignore_index=True)

    recent = usage[usage["month_index"] >= config.months_observed - 3].groupby("customer_id").mean(numeric_only=True)
    earlier = usage[usage["month_index"] < config.months_observed - 3].groupby("customer_id").mean(numeric_only=True)
    active_rate_recent = recent["active_users"].to_numpy() / seats
    active_rate_earlier = earlier["active_users"].to_numpy() / seats
    engagement_drop = active_rate_earlier - active_rate_recent
    tickets_recent = recent["support_tickets"].to_numpy()
    overdue_recent = recent["invoice_overdue"].to_numpy()
    nps_recent = recent["nps"].to_numpy()
    near_renewal = ((renewal_month - end.month) % 12 <= 2).astype(float)

    churn_logit_no_offer = (
        -2.55
        - 2.35 * active_rate_recent
        + 2.15 * engagement_drop
        + 0.22 * tickets_recent
        + 1.45 * overdue_recent
        - 0.15 * nps_recent
        + 0.72 * near_renewal
        + 0.45 * (segments == "SMB")
        - 0.38 * (segments == "Enterprise")
        + 0.30 * (channels == "Outbound")
        + 0.18 * price_sensitivity
        + rng.normal(0, 0.34, n)
    )
    churn_prob_no_offer = sigmoid(churn_logit_no_offer)

    risk_band = pd.qcut(churn_prob_no_offer, q=5, labels=False, duplicates="drop")
    campaign_probability = np.clip(0.05 + 0.10 * np.asarray(risk_band) + rng.uniform(0, 0.08, n), 0.04, 0.62)
    offer_sent = rng.binomial(1, campaign_probability)
    offer_discount_pct = np.where(
        offer_sent == 1,
        rng.choice([10, 15, 20, 25], size=n, p=[0.28, 0.34, 0.26, 0.12]),
        0,
    )
    offer_channel = np.where(
        offer_sent == 1,
        _choice(rng, ["email", "csm_call", "in_app"], [0.46, 0.36, 0.18], n),
        "none",
    )

    treatment_logit_effect = (
        -0.12
        - 1.05 * price_sensitivity * (active_rate_recent > 0.30)
        - 0.62 * near_renewal
        - 0.32 * (offer_discount_pct >= 20)
        + 0.42 * (active_rate_recent < 0.12)
        + 0.20 * (segments == "Enterprise")
        + rng.normal(0, 0.08, n)
    )
    churn_prob = sigmoid(churn_logit_no_offer + offer_sent * treatment_logit_effect)
    churn_next_90d = rng.binomial(1, churn_prob)
    days_until_churn = np.where(churn_next_90d == 1, rng.integers(10, 91, n), 0)
    offer_cost = np.round(np.where(offer_sent == 1, mrr * (offer_discount_pct / 100) * 3, 0), 2)
    realized_90d_revenue = np.round(np.where(churn_next_90d == 1, mrr * days_until_churn / 30, mrr * 3), 2)

    campaign = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "offer_sent": offer_sent,
            "offer_discount_pct": offer_discount_pct,
            "offer_channel": offer_channel,
            "offer_cost": offer_cost,
            "churn_next_90d": churn_next_90d,
            "days_until_churn": days_until_churn,
            "realized_90d_revenue": realized_90d_revenue,
            "true_churn_probability_no_offer": np.round(churn_prob_no_offer, 5),
            "true_churn_probability_observed": np.round(churn_prob, 5),
        }
    )

    paths = {
        "customers": output_dir / "customers.csv",
        "usage": output_dir / "monthly_usage.csv",
        "campaign": output_dir / "retention_campaign.csv",
    }
    customers.to_csv(paths["customers"], index=False)
    usage.to_csv(paths["usage"], index=False)
    campaign.to_csv(paths["campaign"], index=False)
    return paths
