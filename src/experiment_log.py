"""
Experiment Logger for Model 1 (Logistic Regression + TF-IDF).

Consolidates all metrics from aspect detection, sentiment classification,
and hardware report into a single experiment log entry.
Appends to outputs/experiment_log.json.
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: str) -> pd.DataFrame:
    """Load CSV split and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    return df


def compute_aspect_metrics(model, mlb, X, Y_true) -> dict:
    """Compute micro and macro F1 for aspect detection."""
    Y_pred = model.predict(X)
    micro_f1 = round(float(f1_score(Y_true, Y_pred, average="micro")), 4)
    macro_f1 = round(float(f1_score(Y_true, Y_pred, average="macro")), 4)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def compute_sentiment_metrics(sentiment_models, sentiment_vectorizers,
                               df: pd.DataFrame, pipeline: PreprocessingPipeline,
                               aspect_labels: list) -> dict:
    """
    Compute macro and weighted F1 across all per-aspect sentiment classifiers.
    Aggregates predictions across all aspects.
    """
    all_true = []
    all_pred = []

    for aspect in aspect_labels:
        # Filter rows containing this aspect
        mask = df["aspects"].apply(lambda a: aspect in a)
        subset = df[mask]
        if len(subset) == 0:
            continue

        true_labels = subset["aspect_sentiments"].apply(lambda d: d.get(aspect, "neutral")).tolist()

        model = sentiment_models.get(aspect)
        vectorizer = sentiment_vectorizers.get(aspect)

        if model is None or vectorizer is None:
            continue

        # Handle fallback models
        if isinstance(model, dict) and model.get("type") == "fallback":
            pred_labels = [model["prediction"]] * len(true_labels)
        else:
            texts = pipeline.transform(subset["feedback"].tolist())
            X = vectorizer.transform(texts)
            pred_labels = model.predict(X).tolist()

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    if not all_true:
        return {"macro_f1": 0.0, "weighted_f1": 0.0}

    macro_f1 = round(float(f1_score(all_true, all_pred, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(all_true, all_pred, average="weighted", zero_division=0)), 4)
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}


def main():
    logger.info("=" * 70)
    logger.info("EXPERIMENT LOG — Model 1 (LR + TF-IDF)")
    logger.info("=" * 70)

    # ─── Config & Paths ───────────────────────────────────────────────────
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(project_root, "outputs", "experiment_log.json")

    # ─── Load Data Splits ─────────────────────────────────────────────────
    logger.info("Loading data splits...")
    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))

    # ─── Load Aspect Detection Model ─────────────────────────────────────
    logger.info("Loading aspect detection model...")
    aspect_model = joblib.load(os.path.join(output_dir, "aspect_detector_lr.pkl"))
    mlb = joblib.load(os.path.join(output_dir, "mlb.pkl"))
    tfidf = TFIDFFeatures()
    tfidf.load(os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    pipeline = PreprocessingPipeline()
    pipeline.load(os.path.join(output_dir, "preprocessing_pipeline.joblib"))

    # ─── Load Sentiment Models ────────────────────────────────────────────
    logger.info("Loading sentiment classifiers...")
    sentiment_models = joblib.load(os.path.join(output_dir, "sentiment_classifiers_lr.pkl"))
    sentiment_vectorizers = joblib.load(os.path.join(output_dir, "sentiment_vectorizers_lr.pkl"))

    # ─── Load Hardware Report ─────────────────────────────────────────────
    hw_report_path = os.path.join(output_dir, "model1_hardware_report.json")
    with open(hw_report_path, "r") as f:
        hw_report = json.load(f)

    # ─── Compute Aspect Detection Metrics ─────────────────────────────────
    logger.info("Computing aspect detection metrics...")

    train_texts = pipeline.transform(train_df["feedback"].tolist())
    val_texts = pipeline.transform(val_df["feedback"].tolist())
    test_texts = pipeline.transform(test_df["feedback"].tolist())

    X_train = tfidf.transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    Y_train = mlb.transform(train_df["aspects"])
    Y_val = mlb.transform(val_df["aspects"])
    Y_test = mlb.transform(test_df["aspects"])

    aspect_train = compute_aspect_metrics(aspect_model, mlb, X_train, Y_train)
    aspect_val = compute_aspect_metrics(aspect_model, mlb, X_val, Y_val)
    aspect_test = compute_aspect_metrics(aspect_model, mlb, X_test, Y_test)

    logger.info(f"Aspect detection — Train: {aspect_train}, Val: {aspect_val}, Test: {aspect_test}")

    # ─── Compute Sentiment Classification Metrics ─────────────────────────
    logger.info("Computing sentiment classification metrics...")

    sent_train = compute_sentiment_metrics(sentiment_models, sentiment_vectorizers, train_df, pipeline, aspect_labels)
    sent_val = compute_sentiment_metrics(sentiment_models, sentiment_vectorizers, val_df, pipeline, aspect_labels)
    sent_test = compute_sentiment_metrics(sentiment_models, sentiment_vectorizers, test_df, pipeline, aspect_labels)

    logger.info(f"Sentiment — Train: {sent_train}, Val: {sent_val}, Test: {sent_test}")

    # ─── Build Experiment Entry ───────────────────────────────────────────
    entry = {
        "experiment_id": "model1_lr_tfidf",
        "timestamp": datetime.now().isoformat(),
        "model_type": "LogisticRegression + OneVsRest",
        "features": "TF-IDF (10000 features, ngram 1-2)",
        "hyperparameters": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": 42,
        },
        "training_time_seconds": hw_report["training"]["time_seconds"],
        "inference_time_ms_per_sample": hw_report["inference"]["per_sample_avg_ms"],
        "peak_memory_mb": hw_report["training"]["peak_ram_mb"],
        "model_size_mb": hw_report["model_size"]["total_mb"],
        "aspect_detection": {
            "train": aspect_train,
            "val": aspect_val,
            "test": aspect_test,
        },
        "sentiment_classification": {
            "train": sent_train,
            "val": sent_val,
            "test": sent_test,
        },
    }

    # ─── Append to Experiment Log ─────────────────────────────────────────
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            experiment_log = json.load(f)
    else:
        experiment_log = []

    experiment_log.append(entry)

    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)

    logger.info(f"Experiment log saved: {log_path}")

    # ─── Print Summary ────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("EXPERIMENT LOG ENTRY — Model 1")
    print(f"{'═' * 70}")
    print(f"\n  Experiment ID:    {entry['experiment_id']}")
    print(f"  Timestamp:        {entry['timestamp']}")
    print(f"  Model:            {entry['model_type']}")
    print(f"  Features:         {entry['features']}")

    print(f"\n{'─' * 70}")
    print("  ASPECT DETECTION:")
    print(f"  {'Split':<8} {'Micro-F1':>10} {'Macro-F1':>10}")
    print(f"  {'─' * 30}")
    for split in ["train", "val", "test"]:
        m = entry["aspect_detection"][split]
        print(f"  {split:<8} {m['micro_f1']:>10.4f} {m['macro_f1']:>10.4f}")

    print(f"\n{'─' * 70}")
    print("  SENTIMENT CLASSIFICATION:")
    print(f"  {'Split':<8} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print(f"  {'─' * 32}")
    for split in ["train", "val", "test"]:
        m = entry["sentiment_classification"][split]
        print(f"  {split:<8} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")

    print(f"\n{'─' * 70}")
    print("  RESOURCES:")
    print(f"  Training time:       {entry['training_time_seconds']:.3f}s")
    print(f"  Inference/sample:    {entry['inference_time_ms_per_sample']:.4f} ms")
    print(f"  Peak memory:         {entry['peak_memory_mb']:.3f} MB")
    print(f"  Model size:          {entry['model_size_mb']:.4f} MB")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
