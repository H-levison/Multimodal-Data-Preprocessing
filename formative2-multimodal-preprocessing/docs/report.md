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
_TODO: describe audio collection, preprocessing, augmentation, and the voice verification model + results._

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
