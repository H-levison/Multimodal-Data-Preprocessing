"""
product_recommendation_model.py
----------------------------------
OWNER: Gaju (built on Andrew's merged_dataset.csv)

TODO:
  - Load ../data/processed/merged_dataset.csv (see scripts/data_preprocessing.py
    for how it was built, and docs/feature_definitions.md for column definitions)
  - Train a product recommendation model (Random Forest / Logistic Regression / XGBoost)
    predicting `target_product_category`
  - Evaluate with accuracy / F1 / loss
  - Expose a predict_product(customer_features) -> product_category function
    for use by app/cli_app.py
"""
