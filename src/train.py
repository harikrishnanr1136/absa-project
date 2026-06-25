"""
Training module for ABSA Telecom — Aspect Detection (Multi-Label Classification).

Uses OneVsRestClassifier with LogisticRegression on TF-IDF features to detect
which of the 15 aspects are mentioned in each feedback entry.
"""

import json
import logging
import os
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: str) -> pd.DataFrame:
    """Load a CSV split and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    logger.info(f"Loaded {len(df)} rows from {os.path.basename(path)}")
    return df


def build_target_matrix(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> np.ndarray:
    """
    Build binary target matrix Y of shape (n_samples, 15).

    Each row has 1 in columns corresponding to aspects present in that feedback.

    Args:
        df: DataFrame with parsed 'aspects' column (list of aspect strings)
        mlb: Fitted MultiLabelBinarizer

    Returns:
        Binary numpy array (n_samples, 15)
    """
    Y = mlb.transform(df["aspects"])
    logger.info(f"Target matrix shape: {Y.shape} — avg aspects/sample: {Y.sum(axis=1).mean():.2f}")
    return Y


def main():
    logger.info("=" * 70)
    logger.info("ASPECT DETECTION TRAINING — Multi-Label Classification")
    logger.info("=" * 70)

    # ─── Load Config ──────────────────────────────────────────────────────
    config = load_config()
    seed = config["seed"]
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Aspects: {len(aspect_labels)} classes")
    logger.info(f"Seed: {seed}")

    # ─── Step 1: Load train/val/test splits ───────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 1: Loading data splits")
    logger.info("─" * 70)

    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))

    # ─── Step 2: Preprocessing + TF-IDF Features ─────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 2: Preprocessing and TF-IDF feature extraction")
    logger.info("─" * 70)

    pipeline = PreprocessingPipeline()

    logger.info("Preprocessing training texts...")
    train_texts = pipeline.fit_transform(train_df["feedback"].tolist())
    logger.info("Preprocessing val texts...")
    val_texts = pipeline.transform(val_df["feedback"].tolist())
    logger.info("Preprocessing test texts...")
    test_texts = pipeline.transform(test_df["feedback"].tolist())

    logger.info("Extracting TF-IDF features...")
    tfidf = TFIDFFeatures()
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # ─── Step 3-4: Build target matrix with MultiLabelBinarizer ───────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 3: Building multi-label target matrices")
    logger.info("─" * 70)

    mlb = MultiLabelBinarizer(classes=aspect_labels)
    mlb.fit([aspect_labels])  # Fit with all classes to ensure fixed order

    Y_train = build_target_matrix(train_df, mlb)
    Y_val = build_target_matrix(val_df, mlb)
    Y_test = build_target_matrix(test_df, mlb)

    logger.info(f"Label order: {list(mlb.classes_)}")

    # ─── Step 5: Train OneVsRestClassifier ────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 4: Training OneVsRestClassifier(LogisticRegression)")
    logger.info("─" * 70)

    base_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model = OneVsRestClassifier(base_clf, n_jobs=-1)

    # Step 6: Measure training time and memory
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()
    logger.info(f"Memory before training: {mem_before[1] / (1024*1024):.2f} MB peak")

    start_time = time.time()
    logger.info("Training started...")
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time

    mem_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"Training completed in {train_time:.2f} seconds")
    logger.info(f"Memory after training: {mem_after[1] / (1024*1024):.2f} MB peak")
    logger.info(f"Memory used by training: {(mem_after[1] - mem_before[1]) / (1024*1024):.2f} MB")

    # ─── Step 7: Predict on all splits ────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 5: Generating predictions")
    logger.info("─" * 70)

    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)
    Y_test_pred = model.predict(X_test)

    logger.info(f"Train predictions: {Y_train_pred.shape}")
    logger.info(f"Val predictions:   {Y_val_pred.shape}")
    logger.info(f"Test predictions:  {Y_test_pred.shape}")

    # Quick accuracy summary
    from sklearn.metrics import accuracy_score, f1_score

    train_f1 = f1_score(Y_train, Y_train_pred, average="micro")
    val_f1 = f1_score(Y_val, Y_val_pred, average="micro")
    test_f1 = f1_score(Y_test, Y_test_pred, average="micro")

    train_subset_acc = accuracy_score(Y_train, Y_train_pred)
    val_subset_acc = accuracy_score(Y_val, Y_val_pred)
    test_subset_acc = accuracy_score(Y_test, Y_test_pred)

    logger.info("")
    logger.info("─" * 70)
    logger.info("QUICK METRICS SUMMARY")
    logger.info("─" * 70)
    logger.info(f"  {'Split':<8} {'Micro-F1':>10} {'Subset Acc':>12}")
    logger.info(f"  {'─' * 32}")
    logger.info(f"  {'Train':<8} {train_f1:>10.4f} {train_subset_acc:>12.4f}")
    logger.info(f"  {'Val':<8} {val_f1:>10.4f} {val_subset_acc:>12.4f}")
    logger.info(f"  {'Test':<8} {test_f1:>10.4f} {test_subset_acc:>12.4f}")

    # ─── Step 8-9: Save model and MLB ─────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 6: Saving artifacts")
    logger.info("─" * 70)

    model_path = os.path.join(output_dir, "aspect_detector_lr.pkl")
    mlb_path = os.path.join(output_dir, "mlb.pkl")
    tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.joblib")

    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    joblib.dump(mlb, mlb_path)
    logger.info(f"MultiLabelBinarizer saved: {mlb_path}")

    tfidf.save(tfidf_path)
    logger.info(f"TF-IDF vectorizer saved: {tfidf_path}")

    pipeline.save(pipeline_path)
    logger.info(f"Preprocessing pipeline saved: {pipeline_path}")

    # ─── Final Summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Model: OneVsRestClassifier(LogisticRegression)")
    logger.info(f"  Features: TF-IDF ({X_train.shape[1]} dims)")
    logger.info(f"  Classes: {len(aspect_labels)} aspects")
    logger.info(f"  Training time: {train_time:.2f}s")
    logger.info(f"  Val Micro-F1: {val_f1:.4f}")
    logger.info(f"  Test Micro-F1: {test_f1:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
