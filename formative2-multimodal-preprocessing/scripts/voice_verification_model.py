"""
voice_verification_model.py
------------------------------
OWNER: HonourGod, Voiceprint Verification Model

Strategy
--------
This is a *verification* task (does this audio match the claimed identity?),
not a plain classification task (whose voice is this?). A naive classifier
trained directly on `member` labels can't express "reject" for an unknown
claim. Critically, with only one recording session per member, every row
would trivially satisfy "this belongs to its own member", giving a model
with no negative examples to learn from at all.

Instead we build genuine/impostor *pairs*:
  1. For each member, compute a voiceprint "centroid": the mean of their
     acoustic features across their original (non-augmented) recordings.
  2. For every sample (original + augmented) and every member's centroid,
     create one (sample, claimed_identity) pair. The pair is labeled
     genuine (1) if the sample truly belongs to the claimed member, and
     impostor (0) otherwise. The model input is the absolute difference
     between the sample's features and the claimed centroid: small
     differences should mean "genuine", large differences "impostor".
  3. Pairs are split into train/test grouped by the underlying recording
     (`phrase`), so augmented copies of the same clip never leak across
     the split.

Two candidate models are trained and compared:
  - Logistic Regression: linear baseline, cheap, interpretable.
  - Random Forest: handles nonlinear feature interactions, small data.
  - A simple DNN is deliberately not used: with only 8 recordings per
    member (32 base clips total across the team) a DNN has far too few
    samples to train reliably and would overfit. A shallow model is the
    better fit at this data scale.
The better of the two trained candidates (by F1) is kept.

The module exposes:
- `train_voice_verification_model()` -> trains, evaluates, and persists
  the model + metrics report.
- `verify_voice(claimed_member, audio_features) -> (is_verified, confidence)`
  -> verifies a single feature vector against a claimed identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "audio_features.csv"
MODEL_REPORT_PATH = ROOT / "data" / "processed" / "voice_verification_model.json"
MODEL_ARTIFACT_PATH = ROOT / "data" / "processed" / "voice_verification_model.pkl"

FEATURE_COLS = [
    "mfcc_mean",
    "mfcc_std",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "zero_crossing_rate_mean",
    "rms_mean",
    "duration_sec",
]
DIFF_COLS = [f"diff_{col}" for col in FEATURE_COLS]

AUGMENTATION_SUFFIXES = ("_pitch_shift", "_time_stretch", "_background_noise")


def load_feature_data(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("No audio feature rows found. Run scripts/audio_preprocessing.py first.")
    return df


def compute_member_centroids(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Voiceprint reference per member, built only from original (non-augmented)
    recordings so the enrollment reference isn't diluted by synthetic variants."""
    is_original = ~df["sample_name"].str.endswith(AUGMENTATION_SUFFIXES)
    originals = df[is_original]
    return {
        member: group[FEATURE_COLS].mean().to_numpy(dtype=float)
        for member, group in originals.groupby("member")
    }


def build_verification_pairs(df: pd.DataFrame, centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in df.iterrows():
        true_member = sample["member"]
        sample_vec = sample[FEATURE_COLS].to_numpy(dtype=float)
        for claimed_member, centroid in centroids.items():
            diff = np.abs(sample_vec - centroid)
            row = {diff_col: value for diff_col, value in zip(DIFF_COLS, diff)}
            row.update(
                {
                    "phrase": sample["phrase"],
                    "sample_name": sample["sample_name"],
                    "true_member": true_member,
                    "claimed_member": claimed_member,
                    "is_genuine": int(claimed_member == true_member),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def train_voice_verification_model(path: Path = FEATURES_CSV) -> dict[str, Any]:
    df = load_feature_data(path)
    centroids = compute_member_centroids(df)
    pairs = build_verification_pairs(df, centroids)

    X = pairs[DIFF_COLS]
    y = pairs["is_genuine"]
    groups = pairs["phrase"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200, max_depth=5, class_weight="balanced", random_state=42
            ),
        ),
    }

    candidate_reports: dict[str, Any] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        candidate_reports[name] = _evaluate(pipeline, X_test, y_test)

    best_name = max(candidate_reports, key=lambda n: candidate_reports[n]["f1_score"])
    best_pipeline = candidates[best_name]

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "centroids": centroids, "feature_cols": FEATURE_COLS},
        MODEL_ARTIFACT_PATH,
    )

    report = {
        "strategy": "genuine/impostor pairs (|sample_features - claimed_member_centroid|)",
        "models_considered": ["logistic_regression", "random_forest"],
        "dnn_excluded_reason": (
            "Only 8 recordings per member (32 base clips total); a DNN would have "
            "far too little data to train without severe overfitting."
        ),
        "candidate_results": candidate_reports,
        "selected_model": best_name,
        "selected_model_metrics": candidate_reports[best_name],
        "feature_columns": FEATURE_COLS,
        "n_pairs": int(len(pairs)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_artifact() -> dict[str, Any]:
    if not MODEL_ARTIFACT_PATH.exists():
        train_voice_verification_model()
    return joblib.load(MODEL_ARTIFACT_PATH)


def verify_voice(claimed_member: str, audio_features: dict[str, float]) -> tuple[bool, float]:
    """Verify whether `audio_features` (a single sample's acoustic feature dict,
    using the same keys as FEATURE_COLS) matches the voiceprint claimed by
    `claimed_member`. Returns (is_verified, confidence)."""
    artifact = _load_artifact()
    pipeline = artifact["pipeline"]
    centroids = artifact["centroids"]

    if claimed_member not in centroids:
        raise ValueError(f"Unknown claimed identity: {claimed_member!r}")

    sample_vec = np.array([audio_features.get(col, 0.0) for col in FEATURE_COLS], dtype=float)
    diff = np.abs(sample_vec - centroids[claimed_member]).reshape(1, -1)
    diff_df = pd.DataFrame(diff, columns=DIFF_COLS)

    prediction = pipeline.predict(diff_df)[0]
    probabilities = pipeline.predict_proba(diff_df)[0]
    confidence = float(probabilities[1])  # P(genuine)
    return bool(prediction), confidence


if __name__ == "__main__":
    report = train_voice_verification_model()
    print(json.dumps(report, indent=2))
