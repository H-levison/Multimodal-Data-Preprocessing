"""
product_recommendation_model.py
----------------------------------
OWNER: Gaju — Product Recommendation Model (built on Andrew's merged_dataset.csv)

Trains a multiclass classifier that predicts `target_product_category` from the
merged social + transaction feature set produced by scripts/data_preprocessing.py.

Three candidates are compared (Logistic Regression, Random Forest, and Gradient
Boosting as an XGBoost-style stand-in so we don't add a hard xgboost dependency
unless the environment already has it). The best model by F1 is kept.

Exposes:
- train_product_recommendation_model() -> trains, evaluates, persists model + report
- predict_product(customer_features: dict) -> (product_category: str, confidence: float)
  -> used by app/cli_app.py at Stage 2 of the CLI flow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MERGED_CSV = ROOT / "data" / "processed" / "merged_dataset.csv"
MODEL_REPORT_PATH = ROOT / "data" / "processed" / "product_recommendation_model.json"
MODEL_ARTIFACT_PATH = ROOT / "data" / "processed" / "product_recommendation_model.pkl"

# Columns that must NOT be used as model input (identifiers / leakage / raw label)
NON_FEATURE_COLS = {
    "customer_id",
    "target_product_category",
    "target_product_category_encoded",
}


def load_merged_data(path: Path = MERGED_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Ask Andrew for merged_dataset.csv (from "
            "scripts/data_preprocessing.py) and drop it in data/processed/."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("merged_dataset.csv is empty.")
    return df


def build_label_map(df: pd.DataFrame) -> dict[int, str]:
    """Recover the encoded -> readable category mapping that
    `merged["target_product_category_encoded"] = ...astype('category').cat.codes`
    created in data_preprocessing.py."""
    pairs = df[["target_product_category_encoded", "target_product_category"]].drop_duplicates()
    return dict(zip(pairs["target_product_category_encoded"], pairs["target_product_category"]))


def train_product_recommendation_model(path: Path = MERGED_CSV) -> dict[str, Any]:
    df = load_merged_data(path)
    label_map = build_label_map(df)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # Keep only numeric feature columns (the merge script already one-hot encodes
    # categoricals, but guard against stray object columns creeping in).
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[numeric_feature_cols].fillna(0)
    y = df["target_product_category_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.value_counts().min() >= 2 else None
    )

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
        ),
        "gradient_boosting": make_pipeline(
            StandardScaler(), GradientBoostingClassifier(random_state=42)
        ),
    }

    candidate_reports: dict[str, Any] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)
        candidate_reports[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
            "log_loss": float(log_loss(y_test, probs, labels=sorted(y.unique()))),
        }

    best_name = max(candidate_reports, key=lambda n: candidate_reports[n]["f1_score"])
    best_pipeline = candidates[best_name]

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_cols": numeric_feature_cols, "label_map": label_map},
        MODEL_ARTIFACT_PATH,
    )

    report = {
        "models_considered": list(candidates.keys()),
        "candidate_results": candidate_reports,
        "selected_model": best_name,
        "selected_model_metrics": candidate_reports[best_name],
        "feature_columns": numeric_feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(y.nunique()),
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_artifact() -> dict[str, Any]:
    if not MODEL_ARTIFACT_PATH.exists():
        train_product_recommendation_model()
    return joblib.load(MODEL_ARTIFACT_PATH)


def predict_product(customer_features: dict[str, float]) -> tuple[str, float]:
    """Predict the product category for a single customer's feature dict.
    Missing feature keys default to 0 (safe for one-hot columns the customer
    doesn't have a category for)."""
    artifact = _load_artifact()
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    label_map = artifact["label_map"]

    row = pd.DataFrame([[customer_features.get(col, 0.0) for col in feature_cols]], columns=feature_cols)
    pred_encoded = pipeline.predict(row)[0]
    probs = pipeline.predict_proba(row)[0]
    confidence = float(np.max(probs))
    return label_map.get(int(pred_encoded), f"unknown_category_{pred_encoded}"), confidence


if __name__ == "__main__":
    report = train_product_recommendation_model()
    print(json.dumps(report, indent=2))
