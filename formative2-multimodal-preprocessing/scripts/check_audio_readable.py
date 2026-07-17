"""
check_audio_readable.py

Prints a table of audio files under data/raw/audio and whether `soundfile` can read them,
plus file size, so we can identify corrupted/mislabelled files.
"""
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = ROOT / "data" / "raw" / "audio"

if not RAW_AUDIO_DIR.exists():
    print("No audio raw directory found.")
    raise SystemExit(0)

bad = []
for member_dir in sorted(RAW_AUDIO_DIR.iterdir()):
    if not member_dir.is_dir():
        continue
    for f in sorted(member_dir.glob("*.wav")):
        try:
            with sf.SoundFile(str(f)) as fh:
                sr = fh.samplerate
                frames = fh.frames
            readable = True
        except Exception as e:
            readable = False
            sr = None
            frames = None
        size = f.stat().st_size
        print(f"{f.relative_to(RAW_AUDIO_DIR.parent)} | size={size} bytes | readable={readable} | sr={sr} | frames={frames}")
        if not readable:
            bad.append(str(f))

if bad:
    print()
    print("Unreadable files:")
    for p in bad:
        print(p)
else:
    print()
    print("All files readable by soundfile.")
