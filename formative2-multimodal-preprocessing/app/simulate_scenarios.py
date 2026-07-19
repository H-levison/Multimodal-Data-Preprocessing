"""
app/simulate_scenarios.py
-----------------------------
OWNER: Gaju — Section 6 "System Demonstration" deliverable.

Runs the three required scenarios against app/cli_app.run_transaction():
  1. Unauthorized image attempt          -> should fail at Stage 1
  2. Authorized face, unauthorized voice -> should fail at Stage 3
  3. Full successful end-to-end path     -> should reach Stage 4

Prints a summary table at the end suitable for pasting into the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cli_app import run_transaction

AUDIO_DIR = ROOT / "data" / "raw" / "audio"
IMAGE_DIR = ROOT / "data" / "raw" / "images"
DEMO_DIR = ROOT / "data" / "demo"


def main() -> None:
    scenarios = [
        {
            "name": "Unauthorized image attempt",
            "image_path": str(DEMO_DIR / "images" / "unauthorized_face_01.jpg"),
            "customer_id": 100,
            "audio_path": str(AUDIO_DIR / "andrew" / "andrew_yes_approve.wav"),
            "expected_status": "denied",
        },
        {
            "name": "Authorized face, unauthorized voice",
            "image_path": str(IMAGE_DIR / "andrew" / "andrew_neutral.jpg"),
            "customer_id": 100,
            "audio_path": str(DEMO_DIR / "audio" / "yes_approve.wav"),
            "expected_status": "denied",
        },
        {
            "name": "Full successful end-to-end path",
            "image_path": str(IMAGE_DIR / "andrew" / "andrew_neutral.jpg"),
            "customer_id": 100,
            "audio_path": str(AUDIO_DIR / "andrew" / "andrew_yes_approve.wav"),
            "expected_status": "granted",
        },
    ]

    results = []
    for scenario in scenarios:
        print(f"\n----- Scenario: {scenario['name']} -----")
        outcome = run_transaction(
            scenario["image_path"], scenario["customer_id"], scenario["audio_path"]
        )
        passed = outcome["status"] == scenario["expected_status"]
        results.append({**scenario, "outcome": outcome, "passed": passed})

    print("\n=== Simulation Summary ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"[{mark}] {r['name']}: expected={r['expected_status']}, "
            f"got={r['outcome']['status']}"
        )


if __name__ == "__main__":
    main()
