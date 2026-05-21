# Recruiter Pitch

## One-Minute Summary

RetentionIQ is an end-to-end data science project that predicts B2B SaaS churn, estimates which customers are likely to respond to retention offers, and turns model output into a prioritized customer success queue. I built the full workflow: data simulation, feature engineering, model training, calibration, uplift targeting, drift monitoring, and an offline executive dashboard.

## Why It Is Strong

- It solves a business problem recruiters understand: reducing churn and protecting recurring revenue.
- It goes beyond notebook modeling by producing artifacts a team could use.
- It evaluates the model with ranking, calibration, and profit metrics instead of accuracy alone.
- It includes an uplift layer, showing that targeting decisions require more than risk prediction.
- It is safe to publish because the dataset is synthetic.

## Technical Highlights

- Custom L2 logistic regression with `numpy`.
- Platt probability calibration.
- Temporal holdout validation.
- Permutation feature importance.
- T-learner uplift model for retention offers.
- Population stability index for drift monitoring.
- Static HTML dashboard with no external assets.

## How I Would Explain the Business Impact

The churn model identifies accounts likely to leave. The uplift model estimates whether an intervention is likely to change that outcome. Combining risk, customer value, and uplift creates a retention queue that focuses the customer success team on accounts where outreach is expected to pay off.

## Resume Bullet Ideas

- Built an end-to-end churn and uplift modeling system for a synthetic B2B SaaS retention use case, including feature engineering, calibrated probability modeling, policy optimization, and drift monitoring.
- Implemented custom logistic regression, calibration, ranking metrics, permutation importance, and uplift policy valuation in Python with `numpy` and `pandas`.
- Created recruiter-ready artifacts including a model card, scored customer queue, drift report, and offline dashboard for executive communication.
