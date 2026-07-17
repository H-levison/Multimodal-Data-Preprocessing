"""
convert_unreadable_to_wav.py

Try to load audio files (even if their container/codec is nonstandard) using librosa
and rewrite them as standard PCM WAV files so `soundfile` can read them later.

This script targets files in `data/raw/audio/*/*.wav` that raise errors when
`sf.read()` is used.
"""
from pathlib import Path
import warnings
import shutil

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = ROOT / "data" / "raw" / "audio"


def try_convert(path: Path) -> bool:
    try:
        # Try reading with soundfile first to detect files that are already good
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf.read(path)
        return False
    except Exception:
        pass

    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        print(f"Could not load {path} with librosa: {e}")
        return False

    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, audio, sr, subtype="PCM_16")
    # Replace original file with converted file
    backup = path.with_suffix(path.suffix + ".orig")
    try:
        path.replace(backup)
    except Exception:
        shutil.move(str(path), str(backup))
    tmp.replace(path)
    print(f"Converted {path} -> {path} (backup saved as {backup.name})")
    return True


def main():
    converted = 0
    checked = 0
    if not RAW_AUDIO_DIR.exists():
        print("No audio raw directory found; nothing to convert.")
        return

    for member_dir in sorted(RAW_AUDIO_DIR.iterdir()):
        if not member_dir.is_dir():
            continue
        for f in sorted(member_dir.glob("*.wav")):
            checked += 1
            if try_convert(f):
                converted += 1

    print(f"Checked {checked} files, converted {converted} files.")


if __name__ == "__main__":
    main()
