"""
data_preprocessing_real.py
----------------------------
Andrew's Task 1 pipeline, rebuilt against the ACTUAL course datasets:
    - customer_social_profiles.xlsx
    - customer_transactions.xlsx

Key real-world quirks handled here (discovered via inspection):
  * Social profiles use `customer_id_new` formatted like "A178"; transactions
    use `customer_id_legacy` as a bare integer like 178. The numeric part of
    customer_id_new matches customer_id_legacy -> this is the join key.
  * Social profiles have MULTIPLE rows per customer (one per social platform
    they're active on, plus a few exact duplicate rows) -> must be
    aggregated to one row per customer before merging.
  * Transactions has some missing `customer_rating` values.
  * Target leakage is avoided the same way as before: each customer's most
    recent transaction's product_category is held out as the label; all
    RFM/category features are computed only from earlier transactions.
"""

import numpy as np
import pandas as pd

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_THIS_DIR, "..", "data", "raw")
OUT_DIR = os.path.join(_THIS_DIR, "..", "data", "processed")

# =================================================================
# 1. LOAD
# =================================================================
social_raw = pd.read_excel(f"{RAW_DIR}/customer_social_profiles.xlsx")
txn_raw = pd.read_excel(f"{RAW_DIR}/customer_transactions.xlsx")

print("Raw shapes:", social_raw.shape, txn_raw.shape)

# =================================================================
# 2. CLEAN — customer_social_profiles
# =================================================================
social = social_raw.drop_duplicates().copy()

# Reconcile the two ID schemes: "A178" -> 178 (int), matching customer_id_legacy
social["customer_id"] = social["customer_id_new"].str.extract(r"(\d+)").astype(int)

# Normalize text categoricals
social["social_media_platform"] = social["social_media_platform"].astype("string").str.strip().str.title()
social["review_sentiment"] = social["review_sentiment"].astype("string").str.strip().str.title()

# Map sentiment to a numeric score for averaging (Positive=1, Neutral=0, Negative=-1)
sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
social["sentiment_score"] = social["review_sentiment"].map(sentiment_map)

# No missing numeric values in this dataset, but keep a defensive imputer
# in case future data drops have gaps
for col in ["engagement_score", "purchase_interest_score"]:
    if social[col].isna().any():
        social[col] = social[col].fillna(social[col].median())

print("Cleaned social (pre-aggregation):", social.shape,
      "| unique customers:", social["customer_id"].nunique())

# =================================================================
# 2b. AGGREGATE social profile to ONE ROW PER CUSTOMER
#     (a customer can appear on several platforms)
# =================================================================
def mode_or_nan(s):
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan

social_features = social.groupby("customer_id").agg(
    avg_engagement_score=("engagement_score", "mean"),
    max_engagement_score=("engagement_score", "max"),
    avg_purchase_interest_score=("purchase_interest_score", "mean"),
    avg_sentiment_score=("sentiment_score", "mean"),
    dominant_sentiment=("review_sentiment", mode_or_nan),
    primary_platform=("social_media_platform", mode_or_nan),
    platform_diversity=("social_media_platform", "nunique"),
    social_record_count=("social_media_platform", "count"),
).reset_index()

# =================================================================
# 3. CLEAN — customer_transactions
# =================================================================
txn = txn_raw.drop_duplicates(subset="transaction_id").copy()
txn = txn.rename(columns={"customer_id_legacy": "customer_id"})
txn["product_category"] = txn["product_category"].astype("string").str.strip().str.title()
txn["purchase_date"] = pd.to_datetime(txn["purchase_date"])

# Missing customer_rating -> impute with the median rating WITHIN the same
# product category (ratings can systematically differ by category)
txn["customer_rating"] = txn.groupby("product_category")["customer_rating"].transform(
    lambda s: s.fillna(s.median())
)
txn["customer_rating"] = txn["customer_rating"].fillna(txn["customer_rating"].median())

print("Cleaned transactions:", txn.shape, "| unique customers:", txn["customer_id"].nunique())

# =================================================================
# 4. LEAK-FREE TARGET: hold out each customer's most recent purchase
# =================================================================
txn = txn.sort_values(["customer_id", "purchase_date"])
txn_counts = txn.groupby("customer_id").size()
eligible_customers = txn_counts[txn_counts >= 2].index
txn = txn[txn["customer_id"].isin(eligible_customers)]

last_txn = txn.groupby("customer_id").tail(1).set_index("customer_id")
target = last_txn["product_category"].rename("target_product_category")

txn["_rank_from_end"] = txn.groupby("customer_id").cumcount(ascending=False)
history = txn[txn["_rank_from_end"] > 0].drop(columns="_rank_from_end").reset_index(drop=True)

print(f"Customers with >=2 transactions (usable for leak-free target): {len(eligible_customers)}")

# =================================================================
# 4b. TRANSACTION / RFM FEATURE ENGINEERING (history only)
# =================================================================
REFERENCE_DATE = txn_raw["purchase_date"].max() + pd.Timedelta(days=1)

def mode_or_nan2(s):
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan

txn_features = history.groupby("customer_id").agg(
    transaction_frequency=("transaction_id", "count"),
    total_spend=("purchase_amount", "sum"),
    avg_purchase_amount=("purchase_amount", "mean"),
    avg_customer_rating=("customer_rating", "mean"),
    favorite_category=("product_category", mode_or_nan2),
    category_diversity=("product_category", "nunique"),
    last_purchase_date=("purchase_date", "max"),
    first_purchase_date=("purchase_date", "min"),
).reset_index()

txn_features["recency_days"] = (REFERENCE_DATE - txn_features["last_purchase_date"]).dt.days
txn_features["customer_tenure_days"] = (
    txn_features["last_purchase_date"] - txn_features["first_purchase_date"]
).dt.days
txn_features["purchase_cadence_days"] = (
    txn_features["customer_tenure_days"] / (txn_features["transaction_frequency"] - 1).replace(0, np.nan)
).fillna(txn_features["customer_tenure_days"])

txn_features = txn_features.drop(columns=["last_purchase_date", "first_purchase_date"])

# =================================================================
# 5. MERGE social + transaction features + target
# =================================================================
merged = (
    social_features
    .merge(txn_features, on="customer_id", how="inner")
    .merge(target, on="customer_id", how="inner")
)

print("Merged shape (pre-encoding):", merged.shape)

# =================================================================
# 6. ENCODE categoricals
# =================================================================
merged = pd.get_dummies(
    merged,
    columns=["primary_platform", "dominant_sentiment", "favorite_category"],
    prefix=["platform", "sentiment", "favorite_category"],
)
merged["target_product_category_encoded"] = merged["target_product_category"].astype("category").cat.codes

# =================================================================
# 7. SAVE
# =================================================================
merged.to_csv(f"{OUT_DIR}/merged_dataset.csv", index=False)
print("\nFinal merged_dataset.csv shape:", merged.shape)
print("\nColumns:\n", list(merged.columns))
print("\nTarget distribution:\n", merged["target_product_category"].value_counts())
