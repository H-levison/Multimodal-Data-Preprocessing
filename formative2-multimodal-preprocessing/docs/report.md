# Formative 2: System report

## 1. Approach

The system authenticates a user before it shows a product recommendation. Face recognition runs first. If the face is unknown, access stops there. If the face matches a team member, the product model runs in the background on a customer row from the merged tabular dataset, but the prediction stays hidden. Voice verification then checks that the speaker matches the person identified from the face. Only if voice passes does the CLI print the recommended product.

We put the biometrics around the product call on purpose. Face is a cheap first filter. Voice is a second factor that uses a different modality, so a stolen photo alone is not enough. The product score never appears unless both checks pass, which matches the assignment state machine (face -> prediction -> voice -> display or deny).

## 2. Data merge and feature engineering (Andrew)

See `docs/feature_definitions.md` for the full data dictionary. Short summary:

- Merged `customer_social_profiles` (84 unique customers after aggregating 150 cleaned rows across platforms) with `customer_transactions` (75 unique customers, 150 transactions).
- Built a leakage-free target: each customer's most recent purchase category is the label; RFM and behavioral features use only earlier transactions.
- Final `merged_dataset.csv`: 36 customers and 30 features.

## 3. Image pipeline and facial recognition (Divine)

We collected three expressions per team member (neutral, smile, surprised): 12 selfies for Andrew, Divine, Gaju, and Honour under `data/raw/images/<member>/`. Walkthrough with plots: `notebooks/Task2_Image_Pipeline_Divine.ipynb`.

### Preprocessing and visualization

Every photo goes through an OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) to find and crop the face, then resize to 128x128. That also handles the wide range of upload sizes (about 640px WhatsApp compressions up to 4608px phone shots). A face was found in all 12 photos.

Mean brightness and contrast differ by person (Gaju's crops average about 87 intensity; Honour's about 137). Those differences contribute to the recognition signal, similar to how energy helped separate speakers in the voice model.

### Augmentation

Four transforms per photo (assignment asks for at least two): rotation (+15°), horizontal flip, grayscale, and brightness boost (+40).

### Feature extraction

For each original and augmented sample we save an 8x8 downsampled grayscale embedding (64 values), a 32-bin intensity histogram, and mean/std intensity to `data/processed/image_features.csv` (60 rows = 4 members x 3 expressions x 5 variants).

### Facial recognition model

Unlike voice (one recording per phrase, so we needed genuine/impostor pairs), each member has three base photos here. That is enough for a multiclass classifier on the `member` label. Features are standardized and reduced with PCA. We train Logistic Regression and Random Forest and compare them. The test set holds out the "surprised" expression (and its augmentations) for every member, so the model is scored on an expression it did not train on. Splits stay grouped so an augmented copy of a test photo never appears in training, and classes stay balanced.

A plain classifier always names some known member, so `predict_face()` also applies `UNKNOWN_THRESHOLD = 0.5` on the top class probability. Low confidence, or no detectable face, is treated as unrecognized.

### Evaluation

Held-out grouped split, 40 train / 20 test rows:

| Model | Accuracy | F1 (macro) | Log loss |
|---|---|---|---|
| Logistic Regression (selected) | 0.90 | 0.90 | 0.242 |
| Random Forest | 0.90 | 0.896 | 0.557 |

Logistic Regression won. `predict_face(image_path) -> (is_known_user, predicted_member, confidence)` in `scripts/facial_recognition_model.py` is the face gate used by `app/cli_app.py`. The predicted member is also the identity claim passed to `verify_voice()`. Spot checks: each member's held-out surprised photo is recognized (confidence 0.66 to 0.99), and a synthetic no-face unauthorized image is rejected (`is_known_user=False`, confidence 0.0).

## 4. Audio pipeline and voice verification (HonourGod)

We recorded two phrases per member ("Yes, approve" and "Confirm transaction"): eight clips for Andrew, Divine, Honour, and Gaju under `data/raw/audio/<member>/`. Walkthrough with plots: `notebooks/Task3_Audio_Pipeline_HonourGod.ipynb`.

### Preprocessing and visualization

Recordings are resampled to 16 kHz mono. Waveform and log-frequency spectrogram plots for every sample are in `data/processed/plots/`. RMS energy and spectral centroid differ by speaker (Divine is loudest, RMS about 0.044 to 0.049; Gaju has the highest centroid, about 2,300 to 2,700 Hz). Those cues feed the verification approach below.

### Augmentation

Three transforms per recording (assignment asks for at least two): pitch shift (+2 semitones), time stretch (1.1x), and background noise (seeded Gaussian, σ=0.01) so reruns stay reproducible.

### Feature extraction

MFCC mean/std, spectral centroid, spectral rolloff, zero-crossing rate, RMS energy, and duration go into `data/processed/audio_features.csv` for every original and augmented sample (32 rows = 4 members x 2 phrases x 4 variants).

### Voiceprint verification model

This is verification (does the sample match the claimed identity?), not open-set classification. We train on genuine/impostor pairs. Each member gets an acoustic-feature centroid from their original recordings. Every sample (original and augmented) is paired with every centroid and labeled genuine when the claimed identity matches the true speaker. The model input is the absolute difference between sample features and the claimed centroid. Train/test splits are grouped by source recording so augmented copies of the same clip do not leak across the split.

We compared Logistic Regression and Random Forest. We skipped a DNN: eight base recordings per member (32 clips after augmentation) is too little for a neural net without heavy overfitting.

### Evaluation

Held-out grouped split, 80 train / 48 test pairs (from `data/processed/voice_verification_model.json`):

| Model | Accuracy | F1 | Log loss |
|---|---|---|---|
| Logistic Regression (selected) | 0.979 | 0.960 | 0.182 |
| Random Forest | 0.979 | 0.960 | 0.117 |

Accuracy and F1 tie; the selection rule keeps Logistic Regression when F1 is tied. `verify_voice(claimed_member, audio_features) -> (is_verified, confidence)` in `scripts/voice_verification_model.py` is the voice gate for `app/cli_app.py`. Spot checks: Andrew claimed as Andrew verifies (confidence about 0.67); the same clip claimed as Divine is rejected (confidence about 0.004).

## 5. Model integration and evaluation (Gaju)

`app/cli_app.py` runs four stages in order:

1. Face: `predict_face()` must recognize a known member or access is denied.
2. Product: `predict_product()` runs on a `customer_id` from `merged_dataset.csv`; the result is not shown yet.
3. Voice: `verify_voice()` must match the Stage 1 identity or access is denied.
4. Display: only then is the recommended product printed.

Per-model metrics are in `data/processed/*_model.json`. Evaluation plan: `docs/evaluation_plan.md`. The product model compares Logistic Regression, Random Forest, and Gradient Boosting on Andrew's merged features and keeps the best F1 (`scripts/product_recommendation_model.py`).

| Model | Selected algorithm | Accuracy | F1 | Log loss |
|---|---|---|---|---|
| Facial recognition | Logistic Regression | 0.900 | 0.900 | 0.242 |
| Voice verification | Logistic Regression | 0.979 | 0.960 | 0.182 |
| Product recommendation | Logistic Regression | 0.500 | 0.435 | 2.375 |

Product accuracy is weak because the merged set is small (28 train / 8 test, 5 classes). The biometric stages still block access before any recommendation is shown.

## 6. System simulation

`python -m app.simulate_scenarios` runs three demos:

| Scenario | Expected |
|---|---|
| Unauthorized image (`data/demo/images/`) | Denied at Stage 1 |
| Valid face + unauthorized voice (`data/demo/audio/`) | Denied at Stage 3 |
| Valid face + valid voice + `customer_id=100` | Granted at Stage 4 |

Interactive CLI: `python -m app.cli_app`.

## 7. System demonstration video

Link: TODO

## 8. GitHub repository

Link: https://github.com/H-levison/Multimodal-Data-Preprocessing/

## 9. Team contributions

| Member | Contribution |
|---|---|
| Andrew | Data merge, cleaning, and feature engineering for the product recommendation model (`merged_dataset.csv`, `data_preprocessing.py`, EDA notebook) |
| Divine | Image collection, preprocessing, augmentation, facial recognition model |
| HonourGod | Audio collection, preprocessing, augmentation, voice verification model |
| Gaju | Model integration, evaluation, CLI system simulation, report assembly |
