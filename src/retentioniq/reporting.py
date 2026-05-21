from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _bar_rows(importance: pd.DataFrame, limit: int = 12) -> str:
    top = importance.head(limit).copy()
    max_drop = max(float(top["mean_auc_drop"].max()), 1e-9)
    rows = []
    for _, row in top.iterrows():
        width = max(2.0, 100 * float(row["mean_auc_drop"]) / max_drop)
        rows.append(
            f"""
            <div class="bar-row">
              <span>{html.escape(str(row["feature"]))}</span>
              <div class="bar-track"><div class="bar" style="width:{width:.1f}%"></div></div>
              <strong>{float(row["mean_auc_drop"]):.4f}</strong>
            </div>
            """
        )
    return "\n".join(rows)


def _calibration_svg(calibration_bins: list[dict[str, float]]) -> str:
    points = []
    for row in calibration_bins:
        x = 42 + 236 * row["avg_prediction"]
        y = 278 - 236 * row["observed_rate"]
        points.append(f"{x:.1f},{y:.1f}")
    circles = "\n".join(
        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="4" />' for point in points
    )
    polyline = " ".join(points)
    return f"""
    <svg viewBox="0 0 320 320" role="img" aria-label="Calibration chart">
      <line x1="42" y1="278" x2="278" y2="42" class="diag" />
      <polyline points="{polyline}" class="cal-line" />
      {circles}
      <text x="42" y="302">0</text><text x="266" y="302">1.0</text>
      <text x="8" y="282">0</text><text x="8" y="48">1.0</text>
      <text x="105" y="314">Predicted risk</text>
      <text x="-210" y="18" transform="rotate(-90)">Observed churn</text>
    </svg>
    """


def _table(frame: pd.DataFrame, columns: list[str], money_columns: set[str] | None = None) -> str:
    money_columns = money_columns or set()
    header = "".join(f"<th>{html.escape(col.replace('_', ' ').title())}</th>" for col in columns)
    body_rows = []
    for _, row in frame[columns].iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in money_columns:
                rendered = _money(float(value))
            elif isinstance(value, float):
                rendered = _fmt(value)
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_dashboard(
    metrics: dict[str, object],
    scored: pd.DataFrame,
    importance: pd.DataFrame,
    segment_perf: pd.DataFrame,
    drift_report: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    churn_metrics = metrics["churn_model"]
    baseline_metrics = metrics["heuristic_baseline"]
    uplift_metrics = metrics["uplift_policy"]
    threshold = churn_metrics["optimized_threshold"]

    top_customers = scored.sort_values("retention_priority_score", ascending=False).head(12)
    top_drift = drift_report.head(8)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RetentionIQ Model Report</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #64717d;
      --paper: #fbfcfb;
      --line: #d9e1df;
      --accent: #167c80;
      --accent-2: #b1482f;
      --soft: #eef5f3;
      --gold: #b68121;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    header {{
      padding: 36px clamp(20px, 5vw, 64px) 26px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #f4faf8 100%);
    }}
    main {{ padding: 28px clamp(20px, 5vw, 64px) 56px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4vw, 54px); letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    p {{ margin: 0; color: var(--muted); max-width: 920px; line-height: 1.55; }}
    .grid {{ display: grid; gap: 16px; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin-top: 22px; }}
    .card {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 112px;
    }}
    .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }}
    .metric {{ font-size: 32px; font-weight: 760; margin-top: 8px; }}
    .sub {{ color: var(--muted); margin-top: 6px; font-size: 14px; }}
    section {{ margin-top: 30px; }}
    .two-col {{ grid-template-columns: minmax(0, 1.08fr) minmax(300px, .92fr); align-items: start; }}
    .panel {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      overflow-x: auto;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(190px, 260px) 1fr 72px;
      gap: 12px;
      align-items: center;
      min-height: 30px;
      font-size: 14px;
    }}
    .bar-track {{ background: var(--soft); border-radius: 999px; height: 10px; overflow: hidden; }}
    .bar {{ height: 10px; background: var(--accent); border-radius: 999px; }}
    svg {{ width: 100%; max-height: 320px; }}
    svg line, svg polyline {{ fill: none; stroke-width: 2; }}
    .diag {{ stroke: #b9c8c4; stroke-dasharray: 5 5; }}
    .cal-line {{ stroke: var(--accent-2); }}
    svg circle {{ fill: var(--accent-2); }}
    svg text {{ fill: var(--muted); font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 9px; background: var(--soft); color: var(--accent); font-weight: 700; }}
    .note {{ margin-top: 10px; color: var(--muted); font-size: 14px; }}
    @media (max-width: 880px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 6px; }}
    }}
  </style>
</head>
<body>
  <header>
    <span class="badge">RetentionIQ</span>
    <h1>Churn Risk, Uplift Targeting, and Drift Monitoring</h1>
    <p>An end-to-end data science project for a B2B SaaS retention team: generate realistic operational data, build leakage-safe customer features, train a calibrated churn model, identify customers worth targeting, and monitor population shift.</p>
    <div class="grid cards">
      <div class="card"><div class="label">ROC AUC</div><div class="metric">{_fmt(churn_metrics["roc_auc"])}</div><div class="sub">Baseline: {_fmt(baseline_metrics["roc_auc"])}</div></div>
      <div class="card"><div class="label">PR AUC</div><div class="metric">{_fmt(churn_metrics["average_precision"])}</div><div class="sub">Churn prevalence-aware ranking</div></div>
      <div class="card"><div class="label">Lift @ 10%</div><div class="metric">{_fmt(churn_metrics["lift_at_10_pct"])}x</div><div class="sub">Top decile vs average customer</div></div>
      <div class="card"><div class="label">Best Policy Profit</div><div class="metric">{_money(threshold["profit"])}</div><div class="sub">{threshold["selected_customers"]:,} customers selected</div></div>
      <div class="card"><div class="label">Uplift Net Value</div><div class="metric">{_money(uplift_metrics["expected_net_value"])}</div><div class="sub">Top {int(uplift_metrics["top_fraction"] * 100)}% by offer uplift</div></div>
    </div>
  </header>
  <main>
    <section class="grid two-col">
      <div class="panel">
        <h2>Top Model Drivers</h2>
        {_bar_rows(importance)}
        <div class="note">Permutation importance is measured as the holdout ROC AUC drop after shuffling each feature.</div>
      </div>
      <div class="panel">
        <h2>Calibration</h2>
        {_calibration_svg(churn_metrics["calibration_bins"])}
        <div class="note">ECE: {_fmt(churn_metrics["expected_calibration_error"], 4)}. The dashed line is perfect calibration.</div>
      </div>
    </section>
    <section class="panel">
      <h2>Recommended Retention Queue</h2>
      {_table(top_customers, ["customer_id", "segment_label", "plan_label", "mrr", "predicted_churn_risk", "predicted_offer_uplift", "expected_value_at_risk", "recommended_action"], {"mrr", "expected_value_at_risk"})}
    </section>
    <section class="grid two-col">
      <div class="panel">
        <h2>Segment Performance</h2>
        {_table(segment_perf, ["segment", "customers", "actual_churn_rate", "avg_predicted_risk", "avg_predicted_uplift", "value_at_risk"], {"value_at_risk"})}
      </div>
      <div class="panel">
        <h2>Population Stability</h2>
        {_table(top_drift, ["feature", "psi", "status", "train_mean", "holdout_mean"])}
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def render_model_card(
    metrics: dict[str, object],
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    churn = metrics["churn_model"]
    baseline = metrics["heuristic_baseline"]
    uplift = metrics["uplift_policy"]
    threshold = churn["optimized_threshold"]
    top_features = importance.head(8)["feature"].tolist()
    body = f"""# RetentionIQ Model Card

## Intended Use
This model ranks B2B SaaS customers by probability of churn in the next 90 days and estimates which accounts are likely to benefit from a retention offer.

## Model Details
- Primary model: L2-regularized logistic regression trained with custom `numpy` gradient descent.
- Calibration: Platt scaling on a held-out calibration fold.
- Uplift model: two-model T-learner estimating churn probability under contact vs no contact.
- Validation: temporal holdout by signup cohort.

## Holdout Performance
- ROC AUC: {_fmt(churn["roc_auc"])} vs heuristic baseline {_fmt(baseline["roc_auc"])}
- Average precision: {_fmt(churn["average_precision"])}
- Brier score: {_fmt(churn["brier_score"])}
- Expected calibration error: {_fmt(churn["expected_calibration_error"], 4)}
- Lift at top 10%: {_fmt(churn["lift_at_10_pct"])}x
- Capture at top 10%: {_fmt(churn["capture_at_10_pct"])}

## Business Policy
- Optimized contact threshold: {_fmt(threshold["threshold"], 4)}
- Customers selected: {threshold["selected_customers"]:,}
- Estimated saved revenue: {_money(threshold["saved_revenue"])}
- Estimated contact cost: {_money(threshold["contact_cost"])}
- Estimated net profit: {_money(threshold["profit"])}
- Uplift policy net value: {_money(uplift["expected_net_value"])}

## Most Important Features
{chr(10).join(f"- `{feature}`" for feature in top_features)}

## Risks and Guardrails
- The dataset is synthetic, so the repo is safe to publish but should not be presented as proprietary production data.
- The model is designed for prioritization, not automatic account decisions.
- A real deployment should add fairness review, live A/B testing, human approval for high-value accounts, and retraining triggers tied to drift.
"""
    output_path.write_text(body, encoding="utf-8")


def write_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
