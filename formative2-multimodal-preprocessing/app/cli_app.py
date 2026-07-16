"""
cli_app.py
------------
OWNER: Gaju — Model Integration, Evaluation & System Simulation

TODO — implement the full decision flow described in the assignment diagram:

    Start
      -> Facial Recognition Model  --fail--> Access Denied
      -> Run Product Recommendation Model
      -> Voice Validation Model    --fail--> Access Denied
      -> Display Predicted Product

Should import:
    from scripts.facial_recognition_model import predict_face
    from scripts.voice_verification_model import verify_voice
    from scripts.product_recommendation_model import predict_product

Must support simulating BOTH a valid transaction and an unauthorized attempt
(invalid face / invalid voice), per the assignment's System Demonstration requirement.
"""
