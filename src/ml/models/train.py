"""
PowerGuard AI — Model Trainer

Entry-point script for training the risk classifier end-to-end:
  1. Load raw data
  2. Preprocess
  3. Feature engineering
  4. Train/val/test split
  5. Fit RiskClassifier
  6. Evaluate (accuracy, precision, recall, F1, ROC-AUC)
  7. Save model artifact and metrics to ml/models/artifacts/

Run with:
    python -m src.ml.models.train

To be implemented in Phase 3.
"""

# TODO (Phase 3):
#   def train_pipeline(data_dir: str, output_dir: str, params: dict): ...
#
#   if __name__ == "__main__":
#       train_pipeline(...)
