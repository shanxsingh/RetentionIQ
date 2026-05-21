# Data Card: RetentionIQ Synthetic Dataset

## Dataset Purpose

The dataset supports a public, recruiter-ready machine learning project for B2B SaaS retention. It is synthetic by design, so it can be shared without exposing private customers, product telemetry, or revenue records.

## Tables

`customers.csv`

- Customer metadata: segment, plan, industry, region, acquisition channel, seats, monthly recurring revenue, contract length, renewal month, and customer success ownership model.

`monthly_usage.csv`

- Six monthly snapshots per customer.
- Product engagement: active users, logins, collaboration events, automation runs, API calls.
- Friction signals: support tickets, severe tickets, invoice overdue flags, NPS.

`retention_campaign.csv`

- Historical retention offer assignment.
- Offer discount and channel.
- Churn outcome in the next 90 days.
- Realized revenue and synthetic ground-truth churn probabilities for validation.

## Label

`churn_next_90d` is a binary target indicating whether the customer churned within 90 days after the feature observation window.

## Feature Safety

Features are created only from data available before the prediction window:

- Usage aggregates from the prior six months.
- Recent vs prior three-month trend features.
- Contract and renewal context known at scoring time.
- No future churn fields are included in the training feature matrix.

## Known Limitations

- Synthetic patterns are realistic but not a substitute for a production dataset.
- Historical offer assignment is simulated and should be treated as a simplified causal setting.
- A real deployment would need data quality checks, identity resolution, fairness review, and online experimentation.

## Recommended Production Additions

- Warehouse-backed feature definitions.
- Point-in-time feature store or snapshot tables.
- Experiment tracking and model registry.
- A/B testing for retention actions.
- Monitoring for drift, calibration decay, and policy ROI.
