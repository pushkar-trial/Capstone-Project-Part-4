"""Tests for the D2C Churn Scoring API.

Run with:  pytest tests/ -v

Uses FastAPI's TestClient (built on httpx), which runs the app in-process —
no separate server needs to be started before running these tests.
"""
import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_CUSTOMER = {
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
    "last_visit_days_ago": 24,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_threshold"] == 0.30


def test_predict_valid_customer_returns_expected_shape(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()

    assert body["customer_id"] == "CUST09999"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert isinstance(body["risk_explanation"], str) and len(body["risk_explanation"]) > 0
    assert body["threshold_used"] == 0.30
    # predicted_class must be consistent with churn_probability and the threshold
    expected_class = int(body["churn_probability"] >= body["threshold_used"])
    assert body["predicted_class"] == expected_class


def test_predict_high_risk_profile_flagged_high(client):
    """A customer with long recency, zero frequency, and zero recent engagement
    should score as high risk -- this mirrors the Dormant segment from Part 2,
    which has an 87.2% observed churn rate."""
    dormant_customer = copy.deepcopy(VALID_CUSTOMER)
    dormant_customer.update({
        "customer_id": "CUST_DORMANT_TEST",
        "recency_days": 250,
        "frequency_180d": 0,
        "monetary_180d": 0.0,
        "sessions_30d": 0,
        "last_visit_days_ago": 90,
        "ticket_count_90d": 0,
        "negative_ticket_rate_90d": 0.0,
    })
    response = client.post("/predict", json=dormant_customer)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "high"
    assert body["predicted_class"] == 1
    assert body["churn_probability"] >= 0.30


def test_predict_invalid_input_returns_422(client):
    """Missing required fields and an out-of-range value should both be
    rejected before reaching the model, with a 422 (not a 500)."""
    invalid_customer = copy.deepcopy(VALID_CUSTOMER)
    del invalid_customer["recency_days"]  # missing required field
    invalid_customer["return_rate_180d"] = 1.5  # out of the valid [0,1] range

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_invalid_enum_value_returns_422(client):
    """An unrecognized categorical value (not in the data dictionary's value
    set) should be rejected, not silently passed through to the model."""
    invalid_customer = copy.deepcopy(VALID_CUSTOMER)
    invalid_customer["city_tier"] = "Tier 99"

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422


def test_predict_abandoned_carts_exceeding_cart_adds_returns_422(client):
    """Custom cross-field validator: abandoned_carts_30d cannot exceed cart_adds_30d."""
    invalid_customer = copy.deepcopy(VALID_CUSTOMER)
    invalid_customer["cart_adds_30d"] = 1
    invalid_customer["abandoned_carts_30d"] = 5

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422


def test_batch_predict_returns_one_prediction_per_customer(client):
    customer_a = copy.deepcopy(VALID_CUSTOMER)
    customer_a["customer_id"] = "CUST_A"

    customer_b = copy.deepcopy(VALID_CUSTOMER)
    customer_b["customer_id"] = "CUST_B"
    customer_b["recency_days"] = 5
    customer_b["frequency_180d"] = 8

    response = client.post("/batch_predict", json={"customers": [customer_a, customer_b]})
    assert response.status_code == 200
    body = response.json()

    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    returned_ids = {p["customer_id"] for p in body["predictions"]}
    assert returned_ids == {"CUST_A", "CUST_B"}
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["churn_probability"] <= 1.0


def test_batch_predict_rejects_empty_list(client):
    response = client.post("/batch_predict", json={"customers": []})
    assert response.status_code == 422


def test_batch_predict_rejects_more_than_500_customers(client):
    customers = []
    for i in range(501):
        c = copy.deepcopy(VALID_CUSTOMER)
        c["customer_id"] = f"CUST_{i}"
        customers.append(c)

    response = client.post("/batch_predict", json={"customers": customers})
    assert response.status_code == 422
