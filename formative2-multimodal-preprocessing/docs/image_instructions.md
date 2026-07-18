# Image recording instructions

Place face photos in folders named for each team member under
`data/raw/images/`.

Example folders already present in the repo:

- `data/raw/images/andrew/`
- `data/raw/images/divine/`
- `data/raw/images/gaju/`
- `data/raw/images/honour/`

Filename guidelines (place the JPG files directly inside each member folder):

- `<member>_neutral.jpg`
- `<member>_smile.jpg`
- `<member>_surprised.jpg`

Notes:
- One clear, front-facing selfie per expression is enough — the pipeline
  detects and crops the face automatically.
- Supported format: JPG. Any resolution is fine; every face crop is resized
  to a fixed 128x128 during processing, so raw photo size doesn't matter.

After adding files, run:

```bash
python scripts/image_preprocessing.py
```

This will generate:
- `data/processed/image_features.csv`
- `data/processed/plots/*_augmentations.png`

Then, to train the facial recognition model on top of those features:

```bash
python scripts/facial_recognition_model.py
```

This will generate:
- `data/processed/facial_recognition_model.pkl`
- `data/processed/facial_recognition_model.json`
