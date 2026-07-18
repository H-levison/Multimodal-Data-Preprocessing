"""
facial_recognition_model.py
------------------------------
OWNER: Divine, Facial Recognition Model

Strategy
--------
Each team member has 3 base photos (neutral, smile, surprised), each expanded
to 5 variants (original + 4 augmentations) by `image_preprocessing.py` -> 15
rows per member, 60 rows total. That is enough to train a straightforward
multiclass classifier (predict `member` from the appearance features in
`image_features.csv`), unlike the voice task where every member only had one
recording per phrase and a plain classifier had nothing to reject.

The train/test split is grouped by base photo (`member_expression`), holding
out the "surprised" expression (and its augmentations) for every member as
the test set, and training on "neutral" + "smile". This keeps the split
class-balanced (5 test rows per member) while guaranteeing augmented copies
of the same source photo never appear in both train and test.

Two candidate models are trained and compared, same as the voice model:
  - Logistic Regression: linear baseline.
  - Random Forest: handles nonlinear feature interactions.
Features are standardized and reduced with PCA first, since the raw feature
space (~100 columns: pixel embedding + histogram + stats) is wide relative to
60 rows.

Because a plain classifier always predicts *some* known member, even for a
stranger's face, `predict_face` additionally thresholds the model's top
predicted probability: below UNKNOWN_THRESHOLD the caller is told the face is
not recognized at all, rather than trusting a low-confidence guess. This is
what lets the system reject an "unauthorized attempt" image that doesn't
belong to any team member.

The module exposes:
- `train_facial_recognition_model()` -> trains, evaluates, and persists the
  model + metrics report.
- `predict_face(image_path) -> (is_known_user, predicted_member, confidence)`
  -> runs the full pipeline (face detection -> features -> classification) on
  a single image file and returns whether it matches a known user.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.image_preprocessing import detect_face, extract_features, load_image

FEATURES_CSV = ROOT / "data" / "processed" / "image_features.csv"
MODEL_REPORT_PATH = ROOT / "data" / "processed" / "facial_recognition_model.json"
MODEL_ARTIFACT_PATH = ROOT / "data" / "processed" / "facial_recognition_model.pkl"

NON_FEATURE_COLS = {"member", "sample_name", "expression", "face_detected"}
TEST_EXPRESSION = "surprised"
UNKNOWN_THRESHOLD = 0.5


def load_feature_data(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("No image feature rows found. Run scripts/image_preprocessing.py first.")
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the 'surprised' expression (+ its augmentations) per member as
    the test set; train on 'neutral' + 'smile'. Grouped by base expression, so
    no augmented copy of a test photo leaks into training."""
    is_test = df["expression"] == TEST_EXPRESSION
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, classes: list[str]) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "labels": classes,
    }


def train_facial_recognition_model(path: Path = FEATURES_CSV) -> dict[str, Any]:
    df = load_feature_data(path)
    feature_cols = _feature_columns(df)
    train_df, test_df = _split_train_test(df)

    X_train, y_train = train_df[feature_cols], train_df["member"]
    X_test, y_test = test_df[feature_cols], test_df["member"]
    classes = sorted(df["member"].unique())

    n_components = min(10, len(X_train) - 1, len(feature_cols))
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
            RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        ),
    }

    candidate_reports: dict[str, Any] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        candidate_reports[name] = _evaluate(pipeline, X_test, y_test, classes)

    best_name = max(candidate_reports, key=lambda n: candidate_reports[n]["f1_score"])
    best_pipeline = candidates[best_name]

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_cols": feature_cols, "classes": classes},
        MODEL_ARTIFACT_PATH,
    )

    report = {
        "strategy": "multiclass member classification, grouped split holding out the 'surprised' expression",
        "models_considered": ["logistic_regression", "random_forest"],
        "candidate_results": candidate_reports,
        "selected_model": best_name,
        "selected_model_metrics": candidate_reports[best_name],
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "feature_columns": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": classes,
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_artifact() -> dict[str, Any]:
    if not MODEL_ARTIFACT_PATH.exists():
        train_facial_recognition_model()
    return joblib.load(MODEL_ARTIFACT_PATH)


def predict_face(image_path: Path | str) -> tuple[bool, str | None, float]:
    """Run the full pipeline (face detection -> features -> classification) on
    a single image file. Returns (is_known_user, predicted_member, confidence).
    predicted_member is None and is_known_user is False when either no
    confident match is found or no face is detected at all."""
    artifact = _load_artifact()
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    classes = artifact["classes"]

    raw_image = load_image(Path(image_path))
    face, face_detected = detect_face(raw_image)
    if not face_detected:
        return False, None, 0.0

    features = extract_features(face, member="unknown", sample_name="query", expression="query", face_detected=face_detected)
    feature_vector = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)

    probabilities = pipeline.predict_proba(feature_vector)[0]
    best_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[best_idx])
    predicted_member = classes[best_idx]

    if confidence < UNKNOWN_THRESHOLD:
        return False, None, confidence
    return True, predicted_member, confidence


if __name__ == "__main__":
    report = train_facial_recognition_model()
    print(json.dumps(report, indent=2))
