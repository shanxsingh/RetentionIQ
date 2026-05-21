# RetentionIQ Model Card

## Intended Use
This model ranks B2B SaaS customers by probability of churn in the next 90 days and estimates which accounts are likely to benefit from a retention offer.

## Model Details
- Primary model: L2-regularized logistic regression trained with custom `numpy` gradient descent.
- Calibration: Platt scaling on a held-out calibration fold.
- Uplift model: two-model T-learner estimating churn probability under contact vs no contact.
- Validation: temporal holdout by signup cohort.

## Holdout Performance
- ROC AUC: 0.805 vs heuristic baseline 0.747
- Average precision: 0.211
- Brier score: 0.062
- Expected calibration error: 0.0094
- Lift at top 10%: 3.265x
- Capture at top 10%: 0.327

## Business Policy
- Optimized contact threshold: 0.0953
- Customers selected: 620
- Estimated saved revenue: $311,735
- Estimated contact cost: $194,800
- Estimated net profit: $116,936
- Uplift policy net value: $628,637

## Most Important Features
- `active_rate_change_3m`
- `invoice_overdue_recent_3m`
- `active_rate_recent_3m`
- `active_rate_slope`
- `severe_ticket_share_6m`
- `nps_change_3m`
- `near_renewal`
- `nps_recent_3m`

## Risks and Guardrails
- The dataset is synthetic, so the repo is safe to publish but should not be presented as proprietary production data.
- The model is designed for prioritization, not automatic account decisions.
- A real deployment should add fairness review, live A/B testing, human approval for high-value accounts, and retraining triggers tied to drift.
