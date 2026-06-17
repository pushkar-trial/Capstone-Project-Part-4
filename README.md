# Part 4 — FastAPI Churn Scoring Service

D2C Customer Churn Capstone, Part 4 of 4. Wraps the Gradient Boosting model trained in Part 3 in a FastAPI service with health, single-prediction, and batch-prediction endpoints, request validation, rule-based risk explanations, tests, and a monitoring plan for production use.

## Repository Contents

```
.
├── app/
│   ├── main.py              # FastAPI app: /health, /predict, /batch_predict
│   ├── schemas.py            # Pydantic request/response models + validation rules
│   └── risk_explanation.py   # Rule-based natural-language risk explanations
├── tests/
│   └── test_main.py          # pytest test suite (9 tests) using FastAPI's TestClient
├── data/
│   └── rfm_modeling_snapshot.csv   # Source data (for train_model.py reproducibility only)
├── model.pkl                  # Trained model from Part 3 (sklearn Pipeline: preprocessing + Gradient Boosting)
├── colab_demo.ipynb            # Runs the real API inside Google Colab (see "On Google Colab" below)
├── train_model.py              # Optional script to reproduce model.pkl from source data
├── monitoring_plan.md           # Data drift, prediction drift, outcome tracking, retraining triggers, responsible use
├── Dockerfile                    # Optional containerized deployment
└── requirements.txt
```

## How to Run

### On Google Colab

A FastAPI service is a long-running server, which doesn't map directly onto Colab's cell-based execution model — but `colab_demo.ipynb` runs the **actual, unmodified app** (`app/main.py`) inside a Colab notebook by launching a real `uvicorn` server in a background thread, then sending real HTTP requests to it.

1. Upload `part4_capstone.zip` to the Colab file browser (left sidebar → Files → upload).
2. Open `colab_demo.ipynb` in Colab.
3. Run the cells top to bottom. It installs dependencies, unzips the app files, starts the server, and exercises `/health`, `/predict`, and `/batch_predict` with real requests — then runs the full pytest suite and (optionally) reproduces `model.pkl` from source data.

No code in `app/` is changed for the Colab demo — only the notebook's own setup cells differ from a normal local run.

### Locally


```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### Running with Docker (optional)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Running the tests

```bash
pytest tests/ -v
```

9 tests cover: health check, a valid single prediction, a high-risk profile correctly flagged, missing/out-of-range fields rejected with 422, an invalid enum value rejected, a cross-field validation rule (abandoned carts can't exceed cart adds), batch prediction with multiple customers, an empty batch rejected, and a batch over the 500-customer limit rejected.

### Reproducing model.pkl (optional)

`model.pkl` is already included in this repository. If you need to regenerate it from source data (e.g., after a retraining trigger — see `monitoring_plan.md`):

```bash
python train_model.py
```

This reproduces the exact model from Part 3 (`churn_model.ipynb`) — Gradient Boosting, `n_estimators=100, max_depth=2, learning_rate=0.10` — and prints the validation ROC-AUC (0.8837) for a quick sanity check against the original training run.

## Endpoints

### `GET /health`

Returns service and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_threshold": 0.30
}
```

### `POST /predict`

Scores a single customer. Request body matches a row of `rfm_modeling_snapshot.csv` (minus `snapshot_date`, `churn_next_60d`, `split`).

**Sample request:**
```json
{
  "customer_id": "CUST09999",
  "city_tier": "Tier 1",
  "age_group": "25-34",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Skin Care",
  "marketing_consent": "Yes",
  "recency_days": 95,
  "frequency_180d": 1,
  "monetary_180d": 540.70,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.23,
  "avg_rating_180d": 4.0,
  "category_diversity_180d": 1,
  "ticket_count_90d": 2,
  "negative_ticket_rate_90d": 0.5,
  "avg_resolution_hours_90d": 6.5,
  "days_since_signup": 300,
  "sessions_30d": 2,
  "product_views_30d": 8,
  "cart_adds_30d": 1,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 1,
  "email_opens_30d": 1,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 24
}
```

**Sample response:**
```json
{
  "customer_id": "CUST09999",
  "churn_probability": 0.742,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Elevated churn risk driven by: last order was 95 days ago, low recent order frequency, recent support tickets with negative sentiment.",
  "threshold_used": 0.30
}
```

### `POST /batch_predict`

Scores up to 500 customers in a single request.

**Sample request:**
```json
{
  "customers": [
    { "customer_id": "CUST_A", "...": "..." },
    { "customer_id": "CUST_B", "...": "..." }
  ]
}
```

**Sample response:**
```json
{
  "predictions": [
    { "customer_id": "CUST_A", "churn_probability": 0.742, "predicted_class": 1, "risk_level": "high", "risk_explanation": "...", "threshold_used": 0.30 },
    { "customer_id": "CUST_B", "churn_probability": 0.115, "predicted_class": 0, "risk_level": "low", "risk_explanation": "...", "threshold_used": 0.30 }
  ],
  "count": 2
}
```

## Validation

All categorical fields are restricted to the exact value sets observed in the training data (e.g., `city_tier` ∈ {Tier 1, Tier 2, Tier 3}); all numeric fields have sensible bounds based on the data dictionary (e.g., `return_rate_180d` ∈ [0,1]). A custom cross-field rule additionally enforces `abandoned_carts_30d ≤ cart_adds_30d`. Invalid input returns a `422` with details, never a raw model error. See `app/schemas.py` for the full rule set.

## Model & Threshold

Loads `model.pkl` directly — a single scikit-learn `Pipeline` containing both preprocessing (one-hot encoding + standard scaling) and the Gradient Boosting classifier trained in Part 3. Uses the business-justified decision threshold of **0.30** (not the default 0.5), chosen in Part 3 by minimizing estimated business cost given a ~7.9x asymmetry between missing a real churner and wasting a retention offer. At this threshold, the model catches 91.7% of actual churners on the held-out test set. Full reasoning in Part 3's `model_card.md`.

`recency_days` is the single most influential feature (73.6% of model importance, per Part 3), which is why it dominates the rule-based explanations generated by `app/risk_explanation.py`.

## Try It Yourself — Real Examples From the Dataset

These three payloads are real customers from `data/rfm_modeling_snapshot.csv` (not invented data), chosen to cover a clearly high-risk case, a clearly low-risk case, and a genuinely borderline case. Each can be pasted directly into the `/predict` "Try it out" box at `http://127.0.0.1:8000/docs` after starting the server (`uvicorn app.main:app --reload`). Expected responses are shown alongside each one for quick verification — if your response doesn't match, something in the environment differs from what was validated (see the scikit-learn version note below).

### 1. High risk — `CUST00159` (actual outcome: churned)

No orders in 474 days, zero web sessions, zero recent activity — about as clear a "going dormant" profile as the dataset contains.

```json
{
  "customer_id": "CUST00159",
  "city_tier": "Tier 2",
  "age_group": "35-44",
  "acquisition_channel": "Organic",
  "loyalty_tier": "Silver",
  "preferred_category": "Makeup",
  "marketing_consent": "Yes",
  "recency_days": 474,
  "frequency_180d": 0,
  "monetary_180d": 0.0,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.0,
  "avg_rating_180d": 3.5,
  "category_diversity_180d": 0,
  "ticket_count_90d": 0,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 0.0,
  "days_since_signup": 630,
  "sessions_30d": 0,
  "product_views_30d": 0,
  "cart_adds_30d": 0,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 0,
  "email_opens_30d": 1,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 60
}
```

**Expected response:**
```json
{
  "customer_id": "CUST00159",
  "churn_probability": 0.9216,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Elevated churn risk driven by: no order in 474 days, no orders in the last 180 days, no site/app visit in 60 days, no web/app sessions in the last 30 days.",
  "threshold_used": 0.3
}
```

### 2. Low risk — `CUST01152` (actual outcome: retained)

Ordered yesterday, 6 orders in the last 180 days, high recent engagement — a healthy, active customer.

```json
{
  "customer_id": "CUST01152",
  "city_tier": "Tier 1",
  "age_group": "25-34",
  "acquisition_channel": "Marketplace",
  "loyalty_tier": "NotEnrolled",
  "preferred_category": "Skin Care",
  "marketing_consent": "No",
  "recency_days": 1,
  "frequency_180d": 6,
  "monetary_180d": 5679.21,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.212,
  "avg_rating_180d": 4.33,
  "category_diversity_180d": 4,
  "ticket_count_90d": 1,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 38.2,
  "days_since_signup": 414,
  "sessions_30d": 9,
  "product_views_30d": 35,
  "cart_adds_30d": 2,
  "wishlist_adds_30d": 1,
  "abandoned_carts_30d": 1,
  "email_opens_30d": 0,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 17
}
```

**Expected response:**
```json
{
  "customer_id": "CUST01152",
  "churn_probability": 0.0405,
  "predicted_class": 0,
  "risk_level": "low",
  "risk_explanation": "Recent order activity and engagement look healthy; no elevated risk signals detected.",
  "threshold_used": 0.3
}
```

### 3. Borderline — `CUST00072` (actual outcome: churned)

Recency right around the 90-day mark, with moderate frequency and decent web activity otherwise — this is the kind of case where the 0.30 business threshold (vs. the default 0.5) actually changes the outcome.

```json
{
  "customer_id": "CUST00072",
  "city_tier": "Tier 1",
  "age_group": "35-44",
  "acquisition_channel": "Organic",
  "loyalty_tier": "NotEnrolled",
  "preferred_category": "Skin Care",
  "marketing_consent": "Yes",
  "recency_days": 90,
  "frequency_180d": 2,
  "monetary_180d": 1697.84,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.21,
  "avg_rating_180d": 2.5,
  "category_diversity_180d": 2,
  "ticket_count_90d": 0,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 0.0,
  "days_since_signup": 343,
  "sessions_30d": 7,
  "product_views_30d": 26,
  "cart_adds_30d": 3,
  "wishlist_adds_30d": 2,
  "abandoned_carts_30d": 2,
  "email_opens_30d": 5,
  "campaign_clicks_30d": 1,
  "last_visit_days_ago": 26
}
```

**Expected response:**
```json
{
  "customer_id": "CUST00072",
  "churn_probability": 0.411,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Elevated churn risk driven by: last order was 90 days ago.",
  "threshold_used": 0.3
}
```

Note this customer's probability (0.411) is below the default 0.5 threshold but above the business-justified 0.30 threshold used in this API — at the default threshold this customer would have been missed (predicted_class=0), but at 0.30 they're correctly flagged, and they did in fact churn. This is a direct, concrete illustration of the threshold-selection reasoning in Part 3's `model_card.md`.

### A note on reproducing these exact numbers

These probabilities were generated using scikit-learn 1.8.0 (the version `model.pkl` was originally trained with). If your installed scikit-learn version differs, you may see an `InconsistentVersionWarning` and, in some version gaps, an outright load error. `requirements.txt` pins `scikit-learn==1.8.0` to avoid this — if you still see a mismatch, either confirm `pip show scikit-learn` reports `1.8.0`, or run `python train_model.py` to regenerate `model.pkl` against whatever version you have installed (probabilities will differ very slightly but the same three customers should still land in the same high/low/borderline categories).

## Responsible Use

This API is a retention-prioritization tool, not an automated decision system. Per the validated metrics, roughly 1 in 4 customers flagged high-risk will not actually churn, and roughly 1 in 11 actual churners will be missed — a human-reviewed retention process (per Part 2's segmentation and budget strategy) should sit between this API's output and any customer-facing action. Full guidance, including data-drift monitoring and retraining triggers, is in `monitoring_plan.md`.
