# Formative 2 — System Report

## 1. Approach
_Team: describe the overall system design — facial recognition -> product recommendation -> voice verification -> display/deny, and why this order/architecture was chosen._

## 2. Data Merge & Feature Engineering (Andrew)
See `docs/feature_definitions.md` for the full data dictionary. Summary:
- Merged `customer_social_profiles` (84 unique customers, aggregated from 150 cleaned rows across multiple platforms) with `customer_transactions` (75 unique customers, 150 transactions).
- Built a leakage-free target: each customer's most recent purchase category is held out as the label; all RFM/behavioral features are computed only from earlier transactions.
- Final `merged_dataset.csv`: 36 customers x 30 features.

## 3. Image Pipeline & Facial Recognition (Divine)
We collected 3 expressions per team member (neutral, smile, surprised) — 12 clean selfies across the 4 team members (Andrew, Divine, Gaju, Honour) — stored under `data/raw/images/<member>/`. Full walkthrough with plots and narrative interpretation: `notebooks/Task2_Image_Pipeline_Divine.ipynb`.

**Preprocessing & visualization**
- Every photo is passed through an OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) to locate and crop the face, resized to a fixed 128x128 — this also normalizes away the large resolution differences in the raw uploads (640px WhatsApp compressions up to 4608px full-resolution phone shots). A face was found in all 12 photos.
- Mean face brightness and contrast show a consistent per-member signature (e.g. Gaju's crops average ~87 intensity vs. Honour's ~137), which is the basis for part of the recognition signal below, the same way per-speaker energy was for the voice model.

**Augmentation** — 4 applied per photo (exceeds the ≥2 requirement): rotation (+15°), horizontal flip, grayscale conversion, and brightness boost (+40).

**Feature extraction** — an 8x8 downsampled grayscale pixel embedding (64 values, a simple appearance descriptor in the spirit of eigenfaces), a 32-bin grayscale intensity histogram, and mean/std intensity, extracted for every original + augmented sample and saved to `data/processed/image_features.csv` (60 rows = 4 members x 3 expressions x 5 variants).

**Facial recognition model — strategy**
Unlike the voice task (one recording per phrase, requiring genuine/impostor pairs), each member has 3 base photos here, which is enough to train a plain multiclass classifier directly on the `member` label. Features are standardized and reduced with PCA, then Logistic Regression and Random Forest are trained and compared. The train/test split holds out the "surprised" expression (+ its augmentations) per member as the test set, so the model is evaluated on an expression it never trained on, while staying grouped (no augmented copy of a test photo leaks into training) and class-balanced.

Because a plain classifier always predicts *some* known member, `predict_face()` also thresholds the model's top predicted probability (`UNKNOWN_THRESHOLD = 0.5`): a low-confidence prediction, or a photo with no detectable face at all, is reported as unrecognized rather than trusted.

**Evaluation** (held-out grouped split, 40 train / 20 test rows):

| Model | Accuracy | F1 (macro) | Log loss |
|---|---|---|---|
| **Logistic Regression (selected)** | **0.90** | **0.90** | **0.242** |
| Random Forest | 0.90 | 0.896 | 0.557 |

Logistic Regression was selected. `predict_face(image_path) -> (is_known_user, predicted_member, confidence)` in `scripts/facial_recognition_model.py` is the function `app/cli_app.py` should call as the face-recognition gate; `predicted_member` also doubles as the identity claim passed to `verify_voice()`. In spot checks, every team member's held-out "surprised" photo is correctly recognized (confidence 0.66–0.99), and a synthetic no-face "unauthorized attempt" image is correctly rejected (`is_known_user=False`, confidence 0.0).

## 4. Audio Pipeline & Voice Verification (HonourGod)
We collected two phrases per team member ("Yes, approve" and "Confirm transaction") — 8 clean recordings across 4 members (Andrew, Divine, Honour, Gaju) — stored under `data/raw/audio/<member>/`. Full walkthrough with plots and narrative interpretation: `notebooks/Task3_Audio_Pipeline_HonourGod.ipynb`.

**Preprocessing & visualization**
- Resampled recordings to 16 kHz mono.
- Plotted waveform + log-frequency spectrogram for every sample, saved to `data/processed/plots/`. Spectral summary stats (RMS energy, spectral centroid) show a clear per-speaker signature — e.g. Divine's recordings are loudest (RMS ≈ 0.044–0.049) and Gaju's have the highest spectral centroid (≈ 2,300–2,700 Hz) — which is the basis for the verification strategy below.

**Augmentation** — 3 applied per recording (exceeds the ≥2 requirement): pitch shift (+2 semitones), time stretch (1.1x), and background noise (seeded Gaussian, σ=0.01, so the pipeline is fully reproducible on rerun).

**Feature extraction** — MFCC mean/std, spectral centroid, spectral rolloff, zero-crossing rate, RMS energy, and duration extracted for every original + augmented sample and saved to `data/processed/audio_features.csv` (32 rows = 4 members × 2 phrases × 4 variants).

**Voiceprint verification model — strategy**
This is a verification task (does a sample match its *claimed* identity?), not plain classification, so the model is trained on genuine/impostor **pairs** rather than raw member labels: each member's acoustic-feature centroid is computed from their original recordings, then every sample (original + augmented) is paired with every member's centroid — labeled genuine if the claimed identity matches the sample's true speaker, impostor otherwise. The model input is `|sample_features − claimed_centroid|`. Train/test splits are grouped by the underlying recording so augmented copies of the same clip never leak across the split.

Logistic Regression and Random Forest were both trained and compared; a simple DNN was deliberately not used — with only 8 recordings per member (32 base clips total) a DNN would have far too little data to avoid severe overfitting.

**Evaluation** (held-out grouped split, 80 train / 48 test pairs):

| Model | Accuracy | F1 | Log loss |
|---|---|---|---|
| Logistic Regression | 0.917 | 0.857 | 0.167 |
| **Random Forest (selected)** | **0.938** | **0.870** | **0.182** |

Random Forest was selected. `verify_voice(claimed_member, audio_features) -> (is_verified, confidence)` in `scripts/voice_verification_model.py` is the function `app/cli_app.py` should call for the voice-verification gate; it correctly accepts genuine claims and rejects impostor claims with low confidence in spot checks (e.g. Andrew's sample claimed as Andrew: verified, confidence 0.67; the same sample claimed as Divine: rejected, confidence 0.004).

## 5. Model Integration & Evaluation (Gaju)
The CLI (`app/cli_app.py`) runs a strict sequential gate:

1. **Face** — `predict_face()` must recognize a known member or access is denied.
2. **Product** — `predict_product()` runs quietly on a `customer_id` from `merged_dataset.csv` (result withheld).
3. **Voice** — `verify_voice()` must match the claimed identity from Stage 1 or access is denied.
4. **Display** — only then is the recommended product shown.

Per-model metrics live in `data/processed/*_model.json`. Evaluation plan: `docs/evaluation_plan.md`. Product recommendation compares Logistic Regression, Random Forest, and Gradient Boosting on Andrew’s merged features and keeps the best by F1 (`scripts/product_recommendation_model.py`).

## 6. System Simulation
`python -m app.simulate_scenarios` runs three demos:

| Scenario | Expected |
|---|---|
| Unauthorized image (`data/demo/images/`) | Denied at Stage 1 |
| Valid face + unauthorized voice (`data/demo/audio/`) | Denied at Stage 3 |
| Valid face + valid voice + `customer_id=100` | Granted at Stage 4 |

Interactive CLI: `python -m app.cli_app`.

## 7. System Demonstration Video
_Link: TODO_

## 8. GitHub Repository
_Link: https://github.com/H-levison/Multimodal-Data-Preprocessing_

## 9. Team Contributions

| Member | Contribution |
|---|---|
| Andrew | Data merge, cleaning, and feature engineering for the Product Recommendation Model (`merged_dataset.csv`, `data_preprocessing.py`, EDA notebook) |
| Divine | Image collection, preprocessing, augmentation, facial recognition model |
| HonourGod | Audio collection, preprocessing, augmentation, voice verification model |
| Gaju | Model integration, evaluation, CLI system simulation, report assembly |
