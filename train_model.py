"""Reproduces model.pkl from data/rfm_modeling_snapshot.csv.

This mirrors the final model selected in Part 3 (churn_model.ipynb) exactly:
a Gradient Boosting classifier (n_estimators=100, max_depth=2, learning_rate=0.10)
on top of one-hot encoding + standard scaling, trained on the 'train' split.

model.pkl is already included in this repository, so running this script is
optional -- it exists for reproducibility, in case the model needs to be
regenerated from the source data (e.g., after a retraining trigger from
monitoring_plan.md).

Usage:
    python train_model.py
"""
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "data/rfm_modeling_snapshot.csv"
OUTPUT_PATH = "model.pkl"

CAT_COLS = [
    "city_tier", "age_group", "acquisition_channel", "loyalty_tier",
    "preferred_category", "marketing_consent",
]
NUM_COLS = [
    "recency_days", "frequency_180d", "monetary_180d", "return_rate_180d",
    "avg_discount_pct_180d", "avg_rating_180d", "category_diversity_180d",
    "ticket_count_90d", "negative_ticket_rate_90d", "avg_resolution_hours_90d",
    "days_since_signup", "sessions_30d", "product_views_30d", "cart_adds_30d",
    "wishlist_adds_30d", "abandoned_carts_30d", "email_opens_30d",
    "campaign_clicks_30d", "last_visit_days_ago",
]
FEATURE_COLS = CAT_COLS + NUM_COLS


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["loyalty_tier"] = df["loyalty_tier"].fillna("NotEnrolled")

    train = df[df.split == "train"]
    val = df[df.split == "validation"]

    X_train, y_train = train[FEATURE_COLS], train["churn_next_60d"]
    X_val, y_val = val[FEATURE_COLS], val["churn_next_60d"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.10, random_state=RANDOM_STATE,
        )),
    ])

    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation ROC-AUC: {val_auc:.4f} (expected ~0.8837, per Part 3)")

    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
