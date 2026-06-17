"""D2C Churn Scoring API.

Loads the trained model from Part 3 (model.pkl — a single sklearn Pipeline
containing both preprocessing and the Gradient Boosting classifier) and
exposes health, single-prediction, and batch-prediction endpoints.

Run with:  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.risk_explanation import explain_risk, risk_level_from_probability
from app.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)

# Business-selected decision threshold from Part 3 (see model_card.md, "Threshold
# Selection & Business Justification"). Chosen by minimizing FN_cost*FN + FP_cost*FP
# on the validation set, given a ~7.9x cost asymmetry favoring recall.
THRESHOLD = 0.30

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

# Exact feature columns + order the model's ColumnTransformer was fit on (Part 3).
# The ColumnTransformer selects by name, so order here doesn't have to match the
# pipeline's internal order, but it must include every one of these columns.
FEATURE_COLUMNS = [
    "city_tier", "age_group", "acquisition_channel", "loyalty_tier",
    "preferred_category", "marketing_consent",
    "recency_days", "frequency_180d", "monetary_180d", "return_rate_180d",
    "avg_discount_pct_180d", "avg_rating_180d", "category_diversity_180d",
    "ticket_count_90d", "negative_ticket_rate_90d", "avg_resolution_hours_90d",
    "days_since_signup", "sessions_30d", "product_views_30d", "cart_adds_30d",
    "wishlist_adds_30d", "abandoned_carts_30d", "email_opens_30d",
    "campaign_clicks_30d", "last_visit_days_ago",
]

# Holds the loaded model so it's shared across requests without reloading from disk.
model_store: Dict[str, object] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_store["model"] = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        # Service still starts so /health can report the problem instead of crashing.
        model_store["model"] = None
    yield
    model_store["model"] = None


app = FastAPI(
    title="D2C Churn Scoring API",
    description=(
        "Scores customers for 60-day churn risk using the Gradient Boosting model "
        "trained in Part 3 of the capstone. See model_card.md for intended use, "
        "limitations, and the rationale behind the 0.30 decision threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 body is already informative; this just keeps the shape
    # consistent ({"detail": ...}) with the rest of the API's error responses.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def _fallback_exception_handler(request: Request, exc: Exception):
    # Catches anything truly unexpected so the API never leaks a raw stack trace.
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {type(exc).__name__}"},
    )


def _customer_to_frame(customer: CustomerFeatures) -> pd.DataFrame:
    """Builds a single-row DataFrame matching the model's expected feature columns."""
    payload = customer.model_dump(exclude={"customer_id"})
    # Enum fields serialize to their string .value via model_dump(mode="python") already
    # being plain str subclasses, but normalize explicitly to be safe.
    for key, value in payload.items():
        if hasattr(value, "value"):
            payload[key] = value.value
    row = {col: payload[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def _build_prediction(customer: CustomerFeatures, probability: float) -> PredictionResponse:
    predicted_class = int(probability >= THRESHOLD)
    risk_level = risk_level_from_probability(probability, THRESHOLD)
    explanation = explain_risk(customer.model_dump(exclude={"customer_id"}), probability)
    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(float(probability), 4),
        predicted_class=predicted_class,
        risk_level=risk_level,
        risk_explanation=explanation,
        threshold_used=THRESHOLD,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model = model_store.get("model")
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_threshold=THRESHOLD if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    model = model_store.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server startup logs.")

    df = _customer_to_frame(customer)
    try:
        probability = float(model.predict_proba(df)[:, 1][0])
    except Exception as exc:  # defensive: surfaces a clean 500 instead of a raw stack trace
        raise HTTPException(status_code=500, detail=f"Model scoring failed: {exc}") from exc

    return _build_prediction(customer, probability)


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictRequest) -> BatchPredictionResponse:
    model = model_store.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server startup logs.")

    frames = [_customer_to_frame(c) for c in request.customers]
    batch_df = pd.concat(frames, ignore_index=True)

    try:
        probabilities = model.predict_proba(batch_df)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model scoring failed: {exc}") from exc

    predictions = [
        _build_prediction(customer, float(prob))
        for customer, prob in zip(request.customers, probabilities)
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
