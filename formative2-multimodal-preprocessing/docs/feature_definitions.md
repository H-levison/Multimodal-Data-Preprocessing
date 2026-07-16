# Feature Definitions — `merged_dataset.csv` (real data)

**Owner:** Andrew (Task 1 — Data Merge & Product Recommendation Feature Engineering)
**Source files:** `customer_social_profiles.xlsx`, `customer_transactions.xlsx` (as provided)
**Grain:** one row per customer
**Final shape:** 36 customers × 30 columns
**Target:** `target_product_category` (+ `target_product_category_encoded`) for the Task 4 Product Recommendation Model

---

## 1. Identifier

| Column | Description |
|---|---|
| `customer_id` | Integer customer key. Reconciled from two different formats in the raw files: `customer_id_new` in social profiles (e.g. `"A178"`) and `customer_id_legacy` in transactions (e.g. `178`) — the numeric part of the former matches the latter. |

## 2. Social engagement features (aggregated from `customer_social_profiles`)

A customer can have multiple rows in the raw file (one per social platform they're active on), so these are aggregated to one row per customer.

| Column | Description |
|---|---|
| `avg_engagement_score` | Mean `engagement_score` across all of the customer's platform records. |
| `max_engagement_score` | Peak engagement score across platforms — captures their single most-active channel. |
| `avg_purchase_interest_score` | Mean of the raw `purchase_interest_score` field across platforms. |
| `avg_sentiment_score` | `review_sentiment` mapped to a numeric scale (Positive=1, Neutral=0, Negative=-1) and averaged. |
| `dominant_sentiment` (one-hot: `sentiment_Positive/Neutral/Negative`) | Most frequent sentiment label for the customer. Ties (e.g. one Positive record and one Negative record) are broken by pandas' default alphabetical ordering of `.mode()`. |
| `primary_platform` (one-hot: `platform_*`) | Most frequent social platform for the customer. Ties are broken the same way — alphabetically, via pandas' default `.mode()` ordering. |
| `platform_diversity` | Count of distinct platforms the customer appears on — a broad social footprint vs. single-platform user. |
| `social_record_count` | Total number of social profile rows for the customer (raw signal of how much social data exists for them). |

## 3. Transaction / RFM features (engineered from `customer_transactions`)

> **Leakage control:** each customer's most recent transaction's `product_category` is held out as the **target**. All features below are computed only from that customer's *earlier* transactions. Customers with fewer than 2 total transactions are excluded (45 of 75 transacting customers qualify).

| Column | Description |
|---|---|
| `transaction_frequency` | Count of prior transactions (Frequency). |
| `total_spend` | Sum of prior `purchase_amount` (Monetary). |
| `avg_purchase_amount` | Mean prior `purchase_amount` (Monetary). |
| `avg_customer_rating` | Mean prior `customer_rating`. |
| `favorite_category` (one-hot: `favorite_category_*`) | Most frequently purchased category in prior history — the strongest naive prior for the next purchase. |
| `category_diversity` | Number of distinct categories purchased previously. |
| `recency_days` | Days between the customer's last prior purchase and the dataset's reference date (day after the latest transaction in the whole file). |
| `customer_tenure_days` | Days between the customer's first and last *prior* purchase. |
| `purchase_cadence_days` | Average days between purchases (`tenure / (frequency − 1)`), a "how often do they buy" signal. |

## 4. Target

| Column | Description |
|---|---|
| `target_product_category` | Product category of the customer's most recent transaction (human-readable). |
| `target_product_category_encoded` | Same label, integer-encoded for convenience. |

---

## Data Quality Issues Found & Handled (real data)

1. **Incompatible ID formats** — social profiles use `"A" + number` (`customer_id_new`), transactions use a bare integer (`customer_id_legacy`). Resolved by extracting the numeric portion of `customer_id_new` and treating it as the same key.
2. **Multiple rows per customer in social profiles** (84 unique customers across 150 cleaned rows — most customers show up on 1–3 platforms) — resolved by aggregating to one row per customer (mean/max/mode as appropriate per column).
3. **5 exact duplicate rows** in the raw social profile file → removed.
4. **Missing `customer_rating`** (10 of 150 transaction rows) → imputed with the median rating **within the same product category**.
5. **Small overlap between sources** — only 84 customers have a social profile and 75 have transactions, with 61 customers in both; after also requiring ≥2 transactions for a leak-free target, the final merged/model-ready dataset is **36 customers**. This is expected given the size of the real files (155 and 150 raw rows respectively) — it is not a bug, but a natural consequence of requiring complete information across all three pieces (social profile + purchase history + held-out label).
