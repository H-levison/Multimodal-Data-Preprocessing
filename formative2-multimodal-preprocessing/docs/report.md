# Formative 2 — System Report

## 1. Approach
_Team: describe the overall system design — facial recognition -> product recommendation -> voice verification -> display/deny, and why this order/architecture was chosen._

## 2. Data Merge & Feature Engineering (Andrew)
See `docs/feature_definitions.md` for the full data dictionary. Summary:
- Merged `customer_social_profiles` (84 unique customers, aggregated from 150 cleaned rows across multiple platforms) with `customer_transactions` (75 unique customers, 150 transactions).
- Built a leakage-free target: each customer's most recent purchase category is held out as the label; all RFM/behavioral features are computed only from earlier transactions.
- Final `merged_dataset.csv`: 36 customers x 30 features.

## 3. Image Pipeline & Facial Recognition (Divine)
_TODO: describe image collection, preprocessing, augmentation, and the facial recognition model + results._

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
_TODO: describe how the three models are combined, evaluation metrics (accuracy/F1/loss) for each, and the multimodal decision logic._

## 6. System Simulation
_TODO: describe/link the CLI demo — one valid transaction walkthrough, one unauthorized attempt (bad face and/or bad voice)._

## 7. System Demonstration Video
_Link: TODO_

## 8. GitHub Repository
_Link: TODO_

## 9. Team Contributions

| Member | Contribution |
|---|---|
| Andrew | Data merge, cleaning, and feature engineering for the Product Recommendation Model (`merged_dataset.csv`, `data_preprocessing.py`, EDA notebook) |
| Divine | Image collection, preprocessing, augmentation, facial recognition model |
| HonourGod | Audio collection, preprocessing, augmentation, voice verification model |
| Gaju | Model integration, evaluation, CLI system simulation, report assembly |
