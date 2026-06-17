# Monitoring Plan — D2C Churn Scoring API

This document covers what to monitor once the churn scoring API (`app/main.py`) is deployed and used operationally by the retention team, expanding on the "Monitoring Needs" section of Part 3's `model_card.md` into concrete, API-focused practices.

---

## 1. Data Drift

**What to monitor:** the distribution of each input feature in real production traffic, compared against the training distribution from `rfm_modeling_snapshot.csv`.

- **Highest priority: `recency_days`.** This feature alone drives ~74% of the model's predictions (Part 3 feature importance). Track its mean, median, and percentiles weekly. A sudden shift (e.g., the average jumping because of a delayed order-data sync) should be treated as a data-pipeline incident, not a genuine change in customer behavior.
- **Secondary: `monetary_180d`, `return_rate_180d`, `frequency_180d`.** Track basic distributional statistics (mean, std, % zero values) on a rolling weekly basis.
- **Categorical drift:** watch the proportion of each `acquisition_channel`, `city_tier`, and `loyalty_tier` value. If the business launches a new acquisition channel or expands into a new city tier not seen in training, the model's encoder will mark it `handle_unknown='ignore'` and effectively zero out that signal — flag immediately, since this silently degrades predictions for that segment without raising an API error.

**Trigger:** if any single numeric feature's weekly mean shifts by more than ~20% from its training-set mean, or a new categorical value appears in production input (which the model will silently encode as "unknown" rather than error on), open a data-quality investigation before trusting predictions for the affected customers.

---

## 2. Prediction Distribution Monitoring

**What to monitor:** the distribution of `churn_probability` and the proportion of customers scored as `predicted_class=1` (i.e., probability ≥ 0.30) across each scoring batch or day.

- **Baseline (from Part 3 test set):** roughly 50% of customers score above the 0.30 threshold at the actual observed test-set churn rate of 50%, but this will vary with the live customer base. Track in production rather than assuming the test-set rate holds going forward.
- **Trigger:** a week-over-week swing of more than ~15 percentage points in the proportion flagged high-risk, without a corresponding known business event (e.g., end of a sale, holiday season), should trigger a check of both the input data pipeline and the model itself.
- Track the **average `churn_probability` per `risk_level` bucket** (low/medium/high) over time — if "low" risk customers start trending toward 0.25-0.29 probabilities consistently, that's an early signal the model's calibration may be drifting even before it crosses the action threshold.

---

## 3. Business Outcome Tracking

**What to monitor:** for every customer scored and acted on (e.g., given a retention offer per Part 2's strategy), track their actual 60-day purchase behavior against the prediction.

- Maintain a log joining `customer_id`, `churn_probability` at scoring time, `predicted_class`, the retention action taken (if any), and the actual outcome 60 days later.
- Compute **realized precision and recall** on a rolling basis (e.g., monthly) and compare against the validation/test metrics reported in `metrics.json` (precision 0.7196, recall 0.9167 at the 0.30 threshold). A meaningful, sustained drop in realized recall is the strongest signal that the model needs retraining.
- Track **campaign cost vs. revenue saved** by segment (tying back into Part 2's budget framework) — this is the metric that ultimately justifies continued investment in the model and the retention program built on top of it.

---

## 4. API Health & Errors

**What to monitor (operational, not model-quality):**

- **`/health` endpoint uptime** — poll on a fixed interval (e.g., every 1-5 minutes) and alert if `model_loaded` is ever `false`, which indicates the model file failed to load on startup (see `app/main.py`'s lifespan handler).
- **422 rate on `/predict` and `/batch_predict`** — a sustained spike in validation errors (e.g., new unexpected categorical values, fields routinely out of range) suggests an upstream system feeding this API has changed its data format and needs to be reconciled with `app/schemas.py`'s expected schema.
- **500 rate** — any 500 response means the model failed to score otherwise-valid input; these should be rare and investigated individually, since they likely indicate a real bug (e.g., a column mismatch) rather than a data-quality issue.
- **Latency** — track `/predict` p50/p95 latency and `/batch_predict` latency as a function of batch size, to catch performance regressions before they affect the retention team's workflow.

---

## 5. Retraining Triggers

Retrain the model (repeating the Part 3 pipeline on a fresh snapshot) when **any** of the following occur:

1. **Scheduled cadence:** at minimum every 90 days, since the underlying customer base, promotional calendar, and acquisition mix will evolve even without an acute trigger.
2. **Realized recall drops meaningfully below the validated 91.7%** (Section 3 above) over a sustained period (e.g., a full month), once enough outcome data has accumulated to measure this reliably.
3. **Sustained data drift** (Section 1) that can't be resolved by fixing an upstream pipeline issue — i.e., the customer base itself has genuinely changed (e.g., a major new acquisition channel ramping up to a significant share of the base).
4. **A new categorical value appears in production** that wasn't in the training data (e.g., a new `acquisition_channel`) and is expected to persist — the model should be retrained to actually learn from it rather than relying on `handle_unknown='ignore'` indefinitely.

---

## Responsible Use — How the Retention Team Should (and Should Not) Use API Output

This expands on Part 3's model card section "When This Model Should Not Be Used," focused specifically on how the API's output should be consumed operationally:

- **Use `churn_probability` and `risk_level` to prioritize and route customers into the segment-based retention strategy from Part 2** — not as a replacement for that strategy. The API tells you *who* is at risk; Part 2's segments and budget framework tell you *what to do about it*.
- **Do not treat the API as a fully automated, no-human-review trigger for customer-facing actions.** At the chosen threshold, roughly 1 in 4 customers flagged high-risk will not actually churn (precision 0.72), and roughly 1 in 11 actual churners will be missed (recall 0.92, meaning an 8.3% miss rate). A human-reviewed campaign process should sit between this score and any action taken, consistent with Part 2's `manual_review_cases.md` approach of treating ambiguous cases individually rather than mechanically.
- **Do not use this API's output for credit, pricing, or account-status decisions.** It was built and validated only for retention-marketing prioritization.
- **Re-validate before using on a different customer population.** If the API is ever pointed at a different country, currency, or product line than the one used for training (D2C beauty/personal care, India, INR), its predictions should not be trusted without retraining and re-evaluation — the feature relationships (especially the dominant `recency_days` signal) were learned from this specific population.
- **Treat the `risk_explanation` field as a communication aid, not a causal proof.** It surfaces the rule-based factors most associated with the prediction (see `app/risk_explanation.py`), informed by the model's feature importance ranking, but it is not a formal attribution method (e.g., SHAP) and should not be over-interpreted as the definitive "reason" for any individual prediction.
