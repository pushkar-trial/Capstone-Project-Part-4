"""Generates a short, human-readable risk explanation for a single prediction.

This is intentionally rule-based rather than a full SHAP/feature-attribution
system, so it stays fast for batch scoring and easy for a non-technical CRM
user to read. The thresholds below are informed by the EDA in Part 1 and the
feature-importance ranking in Part 3 (recency_days dominates at ~74% importance,
followed by monetary_180d and return_rate_180d).
"""
from typing import Any, Dict


def explain_risk(features: Dict[str, Any], churn_probability: float) -> str:
    """Build a short natural-language explanation from the customer's raw features.

    Args:
        features: dict of the customer's input features (as received by the API).
        churn_probability: model's predicted probability of churn.

    Returns:
        A one- or two-sentence explanation string.
    """
    reasons = []

    recency = features.get("recency_days", 0)
    frequency = features.get("frequency_180d", 0)
    last_visit = features.get("last_visit_days_ago", 0)
    neg_ticket_rate = features.get("negative_ticket_rate_90d", 0.0)
    ticket_count = features.get("ticket_count_90d", 0)
    return_rate = features.get("return_rate_180d", 0.0)
    sessions = features.get("sessions_30d", 0)

    # Recency is by far the dominant model driver (Part 3 feature importance: ~74%)
    if recency >= 120:
        reasons.append(f"no order in {recency} days")
    elif recency >= 60:
        reasons.append(f"last order was {recency} days ago")

    if frequency == 0:
        reasons.append("no orders in the last 180 days")
    elif frequency <= 1:
        reasons.append("low recent order frequency")

    if last_visit >= 30:
        reasons.append(f"no site/app visit in {last_visit} days")

    if ticket_count > 0 and neg_ticket_rate >= 0.5:
        reasons.append("recent support tickets with negative sentiment")

    if return_rate >= 0.5:
        reasons.append(f"high return rate ({return_rate:.0%})")

    if sessions == 0 and churn_probability >= 0.5:
        reasons.append("no web/app sessions in the last 30 days")

    if not reasons:
        if churn_probability < 0.30:
            return "Recent order activity and engagement look healthy; no elevated risk signals detected."
        else:
            return "Model flags elevated risk based on the overall feature pattern, though no single dominant factor stands out."

    if churn_probability >= 0.30:
        return "Elevated churn risk driven by: " + ", ".join(reasons) + "."
    else:
        return "Some risk signals present (" + ", ".join(reasons) + "), but overall profile keeps predicted risk below the action threshold."


def risk_level_from_probability(probability: float, threshold: float) -> str:
    """Maps a probability to a 3-level label for CRM-friendly display.

    `threshold` is the model's chosen decision threshold (0.30, see model_card.md
    in Part 3). Anything below threshold/2 is "low", below threshold is "medium",
    and at/above threshold is "high" — keeping the label consistent with the
    actual predicted_class cutoff rather than an arbitrary separate scale.
    """
    if probability >= threshold:
        return "high"
    elif probability >= threshold / 2:
        return "medium"
    else:
        return "low"
