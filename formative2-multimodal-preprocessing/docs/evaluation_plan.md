# Evaluation Plan — Multimodal Authentication & Product Recommendation System

## 1. Per-model evaluation

| Model | Metrics | Where computed |
|---|---|---|
| Facial Recognition (Divine) | Accuracy, F1, validation loss on authorized-vs-unauthorized classification | `facial_recognition_model.py`, saved to `data/processed/facial_recognition_model.json` |
| Voice Verification (HonourGod) | Accuracy, F1, log loss, confusion matrix on genuine-vs-impostor pairs | `voice_verification_model.py` → `data/processed/voice_verification_model.json` (already implemented) |
| Product Recommendation (Gaju) | Accuracy, weighted F1, log loss on multiclass product prediction | `product_recommendation_model.py` → `data/processed/product_recommendation_model.json` (implemented) |

Each model script already follows the same pattern: train 2–3 candidate
algorithms, evaluate on a held-out test split, keep the best by F1, and
write a JSON report. Task 4's job is to **collect and compare these three
JSON reports**, not re-derive the metrics.

## 2. System-level (integration) evaluation

Beyond the three individual models, the *system* is evaluated as a
sequential pipeline:

- **Stage pass-through rate**: of all simulated attempts, how many correctly
  reach each stage vs. get denied at the right stage.
- **End-to-end success rate**: fraction of "should succeed" scenarios that
  actually reach Stage 4 with a product recommendation.
- **False-accept / false-reject at the system level**: an unauthorized user
  who happens to pass Stage 1 (face) but is caught at Stage 3 (voice) is a
  system success even though Stage 1 alone was fooled — this is the point
  of stacking two biometric checks.

`app/simulate_scenarios.py` (built above) produces a pass/fail table for
exactly this — paste its output into the report.

## 3. Required simulation scenarios (Section 6 of the brief)

1. Unauthorized image attempt → must fail at Stage 1.
2. Authorized face + unauthorized voice → must fail at Stage 3.
3. Full authorized end-to-end path → must reach Stage 4 and display a product.

## 4. What goes in the report

- A table of the three models' accuracy / F1 / loss (pulled from their JSON reports).
- The state-machine diagram (already provided in the assignment).
- The simulation summary table (pass/fail per scenario).
- A short discussion of failure modes: e.g. what happens if the face model
  is confident but wrong, or if only one biometric factor is compromised.
