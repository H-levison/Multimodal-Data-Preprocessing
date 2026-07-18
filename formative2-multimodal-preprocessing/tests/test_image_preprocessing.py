import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.image_preprocessing import process_image_directory


class ImagePreprocessingTest(unittest.TestCase):
    def test_process_image_directory_creates_features_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            input_dir = temp_path / "images"
            output_dir = temp_path / "processed"
            input_dir.mkdir(parents=True, exist_ok=True)

            # a plain gray square stands in for a face crop; no real face is
            # needed to exercise the pipeline's augmentation/feature-extraction path
            face = np.full((200, 200, 3), 128, dtype=np.uint8)
            image_path = input_dir / "member_1" / "member_1_neutral.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), face)

            csv_path = output_dir / "image_features.csv"
            plots_dir = output_dir / "plots"

            processed_rows = process_image_directory(input_dir, csv_path, plots_dir)

            self.assertGreaterEqual(len(processed_rows), 5)
            self.assertTrue(csv_path.exists())
            self.assertTrue(plots_dir.exists())
            self.assertGreaterEqual(len(list(plots_dir.glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
