"""
image_preprocessing.py
------------------------
OWNER: Divine, Image Data Collection & Facial Recognition Pipeline

This script:
  - loads face images from data/raw/images/<member>/<member>_<expression>.jpg
  - detects the face region with a Haar cascade and crops to it
  - applies augmentations (rotation, horizontal flip, grayscale, brightness)
  - extracts appearance features (pixel embedding, intensity histogram,
    basic stats) and saves them to data/processed/image_features.csv
  - saves a side-by-side preview plot per member to data/processed/plots/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIR = ROOT / "data" / "raw" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_OUTPUT = PROCESSED_DIR / "image_features.csv"
PLOTS_DIR = PROCESSED_DIR / "plots"

FACE_SIZE = 128           # square crop each detected face is resized to
EMBEDDING_SIZE = 8        # face is downsampled to EMBEDDING_SIZE x EMBEDDING_SIZE for the pixel embedding
HIST_BINS = 32

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load an image as BGR (OpenCV convention)."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def detect_face(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Detect the largest face in `image` and return a square BGR crop resized
    to FACE_SIZE x FACE_SIZE, plus whether a face was actually found. Falls
    back to a centered square crop of the whole image when no face is
    detected (e.g. an unauthorized/non-face input)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        h, w = image.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        crop = image[top:top + side, left:left + side]
        found = False
    else:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])  # largest detected face box
        crop = image[y:y + h, x:x + w]
        found = True

    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE)), found


def apply_augmentations(face: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Return a list of (name, augmented_face) pairs, each the same size as `face`."""
    h, w = face.shape[:2]
    augmented = []

    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(face, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(("rotated", rotated))

    flipped = cv2.flip(face, 1)
    augmented.append(("flipped", flipped))

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    augmented.append(("grayscale", grayscale))

    brightened = cv2.convertScaleAbs(face, alpha=1.0, beta=40)
    augmented.append(("brightened", brightened))

    return augmented


def extract_features(face: np.ndarray, member: str, sample_name: str, expression: str, face_detected: bool) -> Dict[str, Any]:
    """Build a feature row for one face crop: a small flattened pixel embedding
    (a simple appearance descriptor, in the spirit of eigenfaces), a grayscale
    intensity histogram, and a couple of basic image statistics."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    embedding_img = cv2.resize(gray, (EMBEDDING_SIZE, EMBEDDING_SIZE))
    embedding = (embedding_img.astype(np.float32) / 255.0).flatten()

    hist = cv2.calcHist([gray], [0], None, [HIST_BINS], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    row: Dict[str, Any] = {
        "member": member,
        "sample_name": sample_name,
        "expression": expression,
        "face_detected": face_detected,
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
    }
    row.update({f"embedding_{i}": float(v) for i, v in enumerate(embedding)})
    row.update({f"hist_{i}": float(v) for i, v in enumerate(hist)})
    return row


def plot_variants(variants: List[Tuple[str, np.ndarray]], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3.2))
    for ax, (name, img) in zip(axes, variants):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_image_directory(
    input_dir: Path | str = RAW_IMAGES_DIR,
    output_csv: Path | str = CSV_OUTPUT,
    output_plots_dir: Path | str = PLOTS_DIR,
) -> List[Dict[str, Any]]:
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
        for image_file in sorted(member_dir.glob("*.jpg")):
            # filenames are <member>_<expression>.jpg, e.g. andrew_neutral.jpg
            expression = image_file.stem.split("_", 1)[1] if "_" in image_file.stem else image_file.stem
            sample_name = image_file.stem

            try:
                raw_image = load_image(image_file)
            except ValueError as e:
                print(f"Warning: could not read {image_file}: {e}")
                continue

            face, face_detected = detect_face(raw_image)
            rows.append(extract_features(face, member_name, sample_name, expression, face_detected))

            variants = [("original", face)] + apply_augmentations(face)
            plot_variants(
                variants,
                output_plots_dir / f"{member_name}_{expression}_augmentations.png",
                f"{member_name}/{expression}",
            )

            for aug_name, aug_face in variants[1:]:
                rows.append(
                    extract_features(aug_face, member_name, f"{sample_name}_{aug_name}", expression, face_detected)
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return rows


if __name__ == "__main__":
    process_image_directory()
    print(f"Saved image features to {CSV_OUTPUT}")
