# RetentionIQ

Customer churn risk, uplift targeting, and model monitoring.

This repo is built to show recruiters that you can take a business problem from raw operational data to a practical decision system. It uses a synthetic B2B SaaS retention scenario so the project is safe to publish, easy to run, and realistic enough to discuss in interviews.

## What This Project Demonstrates

- End-to-end ML workflow: data generation, feature engineering, training, calibration, evaluation, reporting, and monitoring.
- Leakage-safe customer features from monthly product usage, support, billing, contract, and account metadata.
- Custom L2 logistic regression and Platt calibration implemented with `numpy`.
- Business metrics beyond accuracy: lift, capture rate, retention profit, value at risk, and uplift policy value.
- T-learner uplift modeling to separate "high risk" from "worth targeting with an offer."
- Population stability monitoring for train vs holdout drift.
- Recruiter-friendly artifacts: metrics JSON, scored customer queue, model card, drift report, and offline HTML dashboard.

## Project Structure

```text
.
├── src/retentioniq/
│   ├── data.py          # Synthetic SaaS customer, usage, and retention campaign data
│   ├── features.py      # Leakage-safe feature engineering
│   ├── models.py        # Numpy logistic regression, calibration, uplift model
│   ├── metrics.py       # ROC AUC, PR AUC, calibration, lift, policy value
│   ├── evaluation.py    # Baselines, permutation importance, segment analysis
│   ├── monitoring.py    # Population stability index
│   ├── reporting.py     # Static HTML dashboard and model card
│   └── pipeline.py      # End-to-end CLI
├── docs/
│   ├── data_card.md
│   └── recruiter_pitch.md
├── tests/
├── data/
├── artifacts/
└── reports/
```

## Quickstart

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m retentioniq.pipeline run --customers 8000 --seed 42
```

Or use:

```bash
make run
```

Run tests:

```bash
make test
```

## Generated Outputs

After running the pipeline, open:

- `reports/dashboard.html` - executive model report.
- `artifacts/metrics.json` - model, baseline, and uplift metrics.
- `artifacts/model_card.md` - model card with intended use, metrics, and risks.
- `artifacts/feature_importance.csv` - permutation importance on the holdout set.
- `artifacts/drift_report.csv` - population stability index by feature.
- `artifacts/scored_holdout_customers.csv` - prioritized customer queue with recommended actions.
- `artifacts/churn_model.npz` - saved model parameters.

## Methodology

The dataset simulates a subscription software company with customer metadata, six months of usage telemetry, support activity, billing friction, and historical retention outreach. The label is whether a customer churns in the next 90 days.

The modeling pipeline:

1. Builds customer-level features from only information available before the prediction window.
2. Splits train and holdout cohorts by signup date to mimic a temporal validation strategy.
3. Trains a calibrated churn risk model.
4. Compares performance against a transparent heuristic baseline.
5. Uses a T-learner uplift model to estimate which accounts are likely to respond to a retention offer.
6. Produces a prioritized retention queue based on risk, expected value at risk, and predicted offer uplift.
7. Checks feature drift with population stability index.

## Interview Talking Points

- Why ROC AUC is not enough for retention work, and why lift, capture, calibration, and profit matter.
- Why high churn risk does not automatically mean a customer should receive a discount.
- How calibration changes downstream business decisions when probabilities feed policy thresholds.
- How to avoid leakage when creating time-windowed usage features.
- How this would move from synthetic data to production: warehouse feature jobs, experiment tracking, model registry, A/B testing, monitoring, and human approval flows.

## Notes

This project intentionally avoids heavy ML dependencies so it can run anywhere with `numpy` and `pandas`. In a production version, you could swap the custom logistic regression for scikit-learn, XGBoost, LightGBM, or a causal ML library while keeping the same feature, evaluation, and reporting structure.
