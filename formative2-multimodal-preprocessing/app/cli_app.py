"""
app/cli_app.py
-----------------
OWNER: Gaju — Student D: Model Integration, Evaluation & System Simulation

Implements the sequential state machine from the assignment diagram:

    Stage 1 (Face check)   -> Fail: ACCESS DENIED, exit
    Stage 2 (Inference)    -> quietly compute predicted product (not shown yet)
    Stage 3 (Voice check)  -> Fail: ACCESS DENIED, exit
    Stage 4 (Execution)    -> display the predicted product
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.audio_preprocessing import extract_features, load_audio
from scripts.facial_recognition_model import predict_face
from scripts.product_recommendation_model import predict_product
from scripts.voice_verification_model import verify_voice

MERGED_CSV = ROOT / "data" / "processed" / "merged_dataset.csv"


class AccessDenied(Exception):
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"Access denied at {stage}: {reason}")


def _lookup_customer_row(customer_id: int) -> dict[str, Any]:
    df = pd.read_csv(MERGED_CSV)
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"No customer_id={customer_id} found in merged_dataset.csv")
    return row.iloc[0].to_dict()


def stage1_face_check(image_path: str) -> str:
    """Returns the matched member_id, or raises AccessDenied."""
    is_authorized, member_id, confidence = predict_face(image_path)
    print(
        f"[Stage 1] Face check on '{image_path}' -> authorized={is_authorized}, "
        f"member={member_id}, confidence={confidence:.2f}"
    )
    if not is_authorized or member_id is None:
        raise AccessDenied(
            "Stage 1 (Face)", f"unrecognized face (confidence={confidence:.2f})"
        )
    return member_id


def stage2_predict_product(customer_id: int) -> tuple[str, float]:
    """Quietly compute the predicted product. Not shown until Stage 4."""
    customer_row = _lookup_customer_row(customer_id)
    product, confidence = predict_product(customer_row)
    print("[Stage 2] Product prediction computed silently (withheld until Stage 4).")
    return product, confidence


def stage3_voice_check(audio_path: str, claimed_member: str) -> None:
    audio, sr = load_audio(Path(audio_path))
    features = extract_features(
        audio, sr, label="cli_verification", sample_name="cli_sample", member=claimed_member
    )
    is_verified, confidence = verify_voice(claimed_member, features)
    print(
        f"[Stage 3] Voice check for claimed identity '{claimed_member}' -> "
        f"verified={is_verified}, confidence={confidence:.2f}"
    )
    if not is_verified:
        raise AccessDenied(
            "Stage 3 (Voice)", f"voice did not match (confidence={confidence:.2f})"
        )


def stage4_execute(product: str, confidence: float) -> None:
    print(
        f"[Stage 4] ACCESS GRANTED -> Recommended product: {product} "
        f"(confidence={confidence:.2f})"
    )


def run_transaction(image_path: str, customer_id: int, audio_path: str) -> dict[str, Any]:
    """Runs the full state machine end to end."""
    try:
        member_id = stage1_face_check(image_path)
        product, confidence = stage2_predict_product(customer_id)
        stage3_voice_check(audio_path, claimed_member=member_id)
        stage4_execute(product, confidence)
        return {
            "status": "granted",
            "member": member_id,
            "product": product,
            "confidence": confidence,
        }
    except AccessDenied as e:
        print(f"\n*** ACCESS DENIED at {e.stage}: {e.reason} ***\n")
        return {"status": "denied", "stage": e.stage, "reason": e.reason}


def interactive_main() -> None:
    print("=== Multimodal Authentication & Product Recommendation CLI ===")
    image_path = input("Enter image path: ").strip()
    customer_id = int(input("Enter customer_id (for product lookup): ").strip())
    audio_path = input("Enter audio path: ").strip()
    run_transaction(image_path, customer_id, audio_path)


if __name__ == "__main__":
    interactive_main()
