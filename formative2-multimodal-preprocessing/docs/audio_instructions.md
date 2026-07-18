# Audio recording instructions

Place your audio recordings in folders named for each team member under `data/raw/audio/`.

Example folders already present in the repo:

- `data/raw/audio/andrew/`
- `data/raw/audio/divine/`
- `data/raw/audio/honour/`
- `data/raw/audio/gaju/`

Filename guidelines (place the WAV files directly inside each member folder):

- `yes_approve.wav`
- `confirm_transaction.wav`

Notes:
- Filenames do not need to include the member name — the script reads the parent folder to identify the member.
- Supported format: WAV. Files will be resampled to 16 kHz during processing.

After adding files, run:

```bash
python scripts/audio_preprocessing.py
```

This will generate:
- `data/processed/audio_features.csv`
- `data/processed/plots/*.png`
