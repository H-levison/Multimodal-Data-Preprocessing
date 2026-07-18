"""
audio_preprocessing.py
-------------------------
OWNER: HonourGod, Audio Data Collection & Voice Verification Pipeline

This script:
  - loads audio files from data/raw/audio
  - plots waveforms and spectrograms
  - applies augmentations (pitch shift, time stretch, background noise)
  - extracts acoustic features and saves them to data/processed/audio_features.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = ROOT / "data" / "raw" / "audio"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_OUTPUT = PROCESSED_DIR / "audio_features.csv"
PLOTS_DIR = PROCESSED_DIR / "plots"


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
    return audio.astype(np.float32), sr


def apply_augmentations(audio: np.ndarray, sr: int) -> List[tuple[str, np.ndarray]]:
    augmented = []
    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    time_stretched = librosa.effects.time_stretch(audio, rate=1.1)

    if len(time_stretched) != len(audio):
        time_stretched = librosa.resample(time_stretched, orig_sr=sr, target_sr=sr)

    augmented.append(("pitch_shift", pitch_shifted))
    augmented.append(("time_stretch", time_stretched))

    # Seeded RNG so re-running the pipeline reproduces the exact same features/plots.
    noise = np.random.default_rng(42).normal(0, 0.01, size=audio.shape[0])
    augmented.append(("background_noise", audio + noise))
    return augmented


def extract_features(audio: np.ndarray, sr: int, label: str, sample_name: str, member: str) -> Dict[str, Any]:
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    return {
        "member": member,
        "sample_name": sample_name,
        "phrase": label,
        "mfcc_mean": float(np.mean(mfccs)),
        "mfcc_std": float(np.std(mfccs)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zero_crossing_rate_mean": float(np.mean(zero_crossing)),
        "rms_mean": float(np.mean(rms)),
        "duration_sec": float(len(audio) / sr),
    }


def plot_audio(audio: np.ndarray, sr: int, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(f"Waveform - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {title}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_audio_directory(input_dir: Path | str = RAW_AUDIO_DIR, output_csv: Path | str = CSV_OUTPUT, output_plots_dir: Path | str = PLOTS_DIR) -> List[Dict[str, Any]]:
    ensure_directories()
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_plots_dir = Path(output_plots_dir)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if not input_dir.exists():
        return rows

    for member_dir in sorted(input_dir.iterdir()):
        if not member_dir.is_dir():
            continue

        member_name = member_dir.name
        for audio_file in sorted(member_dir.glob("*.wav")):
            # Use the full filename (without extension) as the phrase label so
            # filenames like `yes_approve.wav` or `confirm_transaction.wav`
            # are preserved regardless of underscore placement or member folder name.
            label = audio_file.stem
            sample_name = audio_file.stem
            try:
                audio, sr = load_audio(audio_file)
            except Exception as e:
                print(f"Warning: could not read {audio_file}: {e}")
                continue
            plot_audio(audio, sr, output_plots_dir / f"{member_name}_{audio_file.stem}.png", f"{member_name}/{sample_name}")

            rows.append(extract_features(audio, sr, label, sample_name, member_name))

            for aug_name, aug_audio in apply_augmentations(audio, sr):
                aug_path = output_plots_dir / f"{member_name}_{audio_file.stem}_{aug_name}.png"
                plot_audio(aug_audio, sr, aug_path, f"{member_name}/{sample_name} [{aug_name}]")
                rows.append(
                    extract_features(aug_audio, sr, label, f"{sample_name}_{aug_name}", member_name)
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return rows


if __name__ == "__main__":
    process_audio_directory()
    print(f"Saved audio features to {CSV_OUTPUT}")
