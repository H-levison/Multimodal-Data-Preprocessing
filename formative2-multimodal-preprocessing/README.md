# Formative 2 — Multimodal Data Preprocessing

User Identity & Product Recommendation System: facial recognition + voice verification + product recommendation, integrated into a simulated CLI decision flow.

```
Start -> Facial Recognition -> Product Recommendation -> Voice Validation -> Display Predicted Product
              |fail                                          |fail
              v                                              v
         Access Denied                                 Access Denied
```

## Repo Structure

```
formative2-multimodal-preprocessing/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.xlsx      # provided
│   │   ├── customer_transactions.xlsx         # provided
│   │   ├── images/                            # Divine: raw facial images per member
│   │   └── audio/                              # HonourGod: raw audio samples per member
│   └── processed/
│       ├── merged_dataset.csv                  # Andrew's output (tabular merge)
│       ├── image_features.csv                  # Divine's output
│       └── audio_features.csv                  # HonourGod's output
├── notebooks/
│   ├── Task1_Data_Merge_Feature_Engineering.ipynb      # Andrew
│   ├── Task2_Image_Pipeline_Divine.ipynb               # Divine
│   ├── Task3_Audio_Pipeline_HonourGod.ipynb            # HonourGod
│   └── Task4_Model_Integration_Simulation_Gaju.ipynb   # Gaju
├── scripts/
│   ├── data_preprocessing.py           # Andrew — merge/clean/feature engineering (importable)
│   ├── image_preprocessing.py          # Divine
│   ├── facial_recognition_model.py     # Divine
│   ├── audio_preprocessing.py          # HonourGod
│   ├── voice_verification_model.py     # HonourGod
│   └── product_recommendation_model.py # Gaju
├── app/
│   └── cli_app.py                      # Gaju — integrates all 3 models into the CLI simulation
└── docs/
    ├── feature_definitions.md          # Andrew — data dictionary for merged_dataset.csv
    └── report.md                       # Team — final submission report
```

## Team & Task Ownership

| Task | Owner | Status |
|---|---|---|
| 1. Data Merge & Product Recommendation Feature Engineering | **Andrew** | ✅ Complete — see `notebooks/Task1_Data_Merge_Feature_Engineering.ipynb` |
| 2. Image Data Collection & Facial Recognition Pipeline | Divine | ⬜ TODO |
| 3. Audio Data Collection & Voice Verification Pipeline | HonourGod | ⬜ TODO |
| 4. Model Integration, Evaluation & System Simulation | Gaju | ⬜ TODO |

## Setup

```bash
pip install -r requirements.txt
```

## Task 1 — Data Merge (Andrew) — How to Reproduce

```bash
python scripts/data_preprocessing.py
```
Reads `data/raw/customer_social_profiles.xlsx` and `data/raw/customer_transactions.xlsx`,
cleans and merges them, and writes `data/processed/merged_dataset.csv`.

For the annotated, plotted, walk-through version (recommended for grading), open
`notebooks/Task1_Data_Merge_Feature_Engineering.ipynb` in Google Colab or Jupyter.

See `docs/feature_definitions.md` for what every column in `merged_dataset.csv` means
and why it was engineered that way (including the leak-free target construction).

## Deliverables Checklist (per assignment brief)

- [x] Datasets (`data/raw/`)
- [x] Merged dataset with feature engineering (`data/processed/merged_dataset.csv`)
- [ ] `image_features.csv` (Divine)
- [ ] `audio_features.csv` (HonourGod)
- [x] Python scripts for the pipeline — Task 1 done; Tasks 2–4 pending
- [x] Jupyter notebook — Task 1 done; Tasks 2–4 pending
- [ ] Report describing approach (`docs/report.md` — skeleton in place)
- [ ] System simulation video link
- [ ] GitHub repository link
- [ ] Team member contributions
