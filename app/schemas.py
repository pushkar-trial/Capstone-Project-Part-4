"""Pydantic request/response models for the churn scoring API.

Field bounds below reflect the ranges documented in DATA_DICTIONARY.md and observed
in rfm_modeling_snapshot.csv (Part 1/3). They exist to catch obviously malformed
input (negative counts, impossible percentages) before it ever reaches the model.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CityTier(str, Enum):
    tier_1 = "Tier 1"
    tier_2 = "Tier 2"
    tier_3 = "Tier 3"


class AgeGroup(str, Enum):
    a18_24 = "18-24"
    a25_34 = "25-34"
    a35_44 = "35-44"
    a45_plus = "45+"


class AcquisitionChannel(str, Enum):
    google_search = "Google Search"
    instagram = "Instagram"
    influencer = "Influencer"
    referral = "Referral"
    marketplace = "Marketplace"
    organic = "Organic"


class LoyaltyTier(str, Enum):
    silver = "Silver"
    gold = "Gold"
    platinum = "Platinum"
    not_enrolled = "NotEnrolled"  # use this when the customer has no loyalty_tier on file


class PreferredCategory(str, Enum):
    skin_care = "Skin Care"
    hair_care = "Hair Care"
    makeup = "Makeup"
    fragrance = "Fragrance"
    wellness = "Wellness"
    baby_care = "Baby Care"


class MarketingConsent(str, Enum):
    yes = "Yes"
    no = "No"


class CustomerFeatures(BaseModel):
    """One customer's feature snapshot, in the same shape as a row of
    rfm_modeling_snapshot.csv (minus customer_id, snapshot_date, churn_next_60d, split).
    """

    customer_id: str = Field(..., description="Customer identifier, echoed back in the response.", min_length=1, max_length=64)

    # categorical
    city_tier: CityTier
    age_group: AgeGroup
    acquisition_channel: AcquisitionChannel
    loyalty_tier: LoyaltyTier
    preferred_category: PreferredCategory
    marketing_consent: MarketingConsent

    # numeric — RFM
    recency_days: int = Field(..., ge=0, le=1000, description="Days since the customer's last pre-snapshot order.")
    frequency_180d: int = Field(..., ge=0, le=200, description="Number of orders in the 180 days before the snapshot.")
    monetary_180d: float = Field(..., ge=0, le=1_000_000, description="Total gross spend (INR) in the 180-day window.")
    return_rate_180d: float = Field(..., ge=0.0, le=1.0, description="Proportion of orders returned in the 180-day window.")
    avg_discount_pct_180d: float = Field(..., ge=0.0, le=1.0, description="Average discount fraction across orders in the 180-day window.")
    avg_rating_180d: float = Field(..., ge=1.0, le=5.0, description="Average order rating in the 180-day window.")
    category_diversity_180d: int = Field(..., ge=0, le=10, description="Number of distinct product categories purchased in the 180-day window.")

    # numeric — support
    ticket_count_90d: int = Field(..., ge=0, le=100, description="Number of support tickets raised in the 90 days before the snapshot.")
    negative_ticket_rate_90d: float = Field(..., ge=0.0, le=1.0, description="Proportion of 90-day tickets with negative sentiment.")
    avg_resolution_hours_90d: float = Field(..., ge=0.0, le=500.0, description="Average ticket resolution time (hours); 0 if no tickets.")

    # numeric — tenure
    days_since_signup: int = Field(..., ge=0, le=5000, description="Days from signup_date to the snapshot date.")

    # numeric — web/app activity (30d)
    sessions_30d: int = Field(..., ge=0, le=1000)
    product_views_30d: int = Field(..., ge=0, le=5000)
    cart_adds_30d: int = Field(..., ge=0, le=1000)
    wishlist_adds_30d: int = Field(..., ge=0, le=1000)
    abandoned_carts_30d: int = Field(..., ge=0, le=1000)
    email_opens_30d: int = Field(..., ge=0, le=1000)
    campaign_clicks_30d: int = Field(..., ge=0, le=1000)
    last_visit_days_ago: int = Field(..., ge=0, le=1000, description="Days since the customer's most recent website/app visit.")

    @field_validator("abandoned_carts_30d")
    @classmethod
    def abandoned_not_more_than_cart_adds(cls, v: int, info) -> int:
        cart_adds = info.data.get("cart_adds_30d")
        if cart_adds is not None and v > cart_adds:
            raise ValueError("abandoned_carts_30d cannot exceed cart_adds_30d")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_level: RiskLevel
    risk_explanation: str
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_threshold: Optional[float] = None
