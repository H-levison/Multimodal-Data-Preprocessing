import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.audio_preprocessing import process_audio_directory


class AudioPreprocessingTest(unittest.TestCase):
    def test_process_audio_directory_creates_features_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            input_dir = temp_path / "audio"
            output_dir = temp_path / "processed"
            input_dir.mkdir(parents=True, exist_ok=True)

            sr = 16000
            duration = 1.0
            t = np.arange(0, duration, 1 / sr)
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            audio_path = input_dir / "member_1" / "yes_approve.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, audio, sr)

            csv_path = output_dir / "audio_features.csv"
            plots_dir = output_dir / "plots"

            processed_rows = process_audio_directory(input_dir, csv_path, plots_dir)

            self.assertGreaterEqual(len(processed_rows), 3)
            self.assertTrue(csv_path.exists())
            self.assertTrue(plots_dir.exists())
            self.assertGreaterEqual(len(list(plots_dir.glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
