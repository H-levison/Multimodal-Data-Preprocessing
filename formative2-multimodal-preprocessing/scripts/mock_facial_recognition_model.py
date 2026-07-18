"""
mock_facial_recognition_model.py
------------------------------------
PLACEHOLDER ONLY — stands in for Divine's facial_recognition_model.py until
her real trained model (image_features.csv + SVM/CNN, per the task breakdown)
is delivered.

Why this exists: Task 4 (integration) cannot wait for every teammate to
finish before the CLI/state-machine can be built and tested. This mock
implements the EXACT function signature the real model must expose, so
swapping it in later is a one-line import change in app/cli_app.py:

    from scripts.facial_recognition_model import recognize_face   # real
    from scripts.mock_facial_recognition_model import recognize_face  # mock

CONTRACT Divine's real facial_recognition_model.py must satisfy:

    def recognize_face(image_path: str | Path) -> tuple[bool, str | None, float]:
        '''
        Returns:
            is_authorized: True if the face matches a known team member
            member_id:     the matched member's name/id, or None if unauthorized
            confidence:    float in [0, 1]
        '''

This mock decides "authorized" purely from the filename, so simulation
scenarios stay deterministic and don't require real images yet:
    - a filename containing "unauthorized" or "unknown"  -> rejected
    - any other filename -> treated as a match for the member name embedded
      in the filename (e.g. "andrew_neutral.jpg" -> member "andrew"), or
      "demo_user" if no known member name is found.
"""

from __future__ import annotations

from pathlib import Path

KNOWN_MEMBERS = {"andrew", "divine", "honour", "gaju"}
REJECT_KEYWORDS = ("unauthorized", "unknown", "intruder", "fake")


def recognize_face(image_path: str | Path) -> tuple[bool, str | None, float]:
    name = Path(image_path).stem.lower()

    if any(keyword in name for keyword in REJECT_KEYWORDS):
        return False, None, 0.12  # low confidence, rejected

    for member in KNOWN_MEMBERS:
        if member in name:
            return True, member, 0.94

    # Unrecognized filename pattern: treat as a low-confidence non-match
    return False, None, 0.30


if __name__ == "__main__":
    for sample in ["andrew_neutral.jpg", "unauthorized_face_01.jpg", "gaju_smile.jpg"]:
        print(sample, "->", recognize_face(sample))
