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
├── train_model.py              # Optional script to reproduce model.pkl from source data
├── monitoring_plan.md           # Data drift, prediction drift, outcome tracking, retraining triggers, responsible use
├── Dockerfile                    # Optional containerized deployment
└── requirements.txt
```

## How to Run

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

## Responsible Use

This API is a retention-prioritization tool, not an automated decision system. Per the validated metrics, roughly 1 in 4 customers flagged high-risk will not actually churn, and roughly 1 in 11 actual churners will be missed — a human-reviewed retention process (per Part 2's segmentation and budget strategy) should sit between this API's output and any customer-facing action. Full guidance, including data-drift monitoring and retraining triggers, is in `monitoring_plan.md`.
