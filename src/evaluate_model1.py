"""
Evaluation Script for Model 1 (Logistic Regression + TF-IDF).

Evaluates aspect detection and per-aspect sentiment classification on train/val/test.
Saves metrics to outputs/models/model1_metrics.json in same structure as model2_metrics.json.
"""

import json
import logging
import os
import time
import tracemalloc
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures
from src.evaluate import evaluate_model

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: str) -> pd.DataFrame:
    """Load CSV and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    return df


def evaluate_aspect_detection(model, tfidf, pipeline, df, mlb, aspect_labels) -> dict:
    """Evaluate aspect detection model on a split."""
    texts = pipeline.transform(df["feedback"].tolist())
    X = tfidf.transform(texts)
    Y_true = mlb.transform(df["aspects"])
    Y_pred = model.predict(X)

    h_loss = float(hamming_loss(Y_true, Y_pred))
    micro_p = float(precision_score(Y_true, Y_pred, average="micro", zero_division=0))
    micro_r = float(recall_score(Y_true, Y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    macro_p = float(precision_score(Y_true, Y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(Y_true, Y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))

    per_aspect_f1 = {}
    for j, aspect in enumerate(aspect_labels):
        per_aspect_f1[aspect] = round(float(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0)), 4)

    return {
        "hamming_loss": round(h_loss, 4),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "per_aspect_f1": per_aspect_f1,
    }


def evaluate_sentiment(sentiment_models, sentiment_vectorizers, df, aspect_labels) -> dict:
    """Evaluate per-aspect sentiment classifiers on a split."""
    all_true = []
    all_pred = []
    per_aspect = {}

    for aspect in aspect_labels:
        mask = df["aspects"].apply(lambda a: aspect in a)
        subset = df[mask]
        if len(subset) == 0:
            continue

        true_labels = subset["aspect_sentiments"].apply(lambda d: d.get(aspect, "neutral")).tolist()

        model = sentiment_models.get(aspect)
        vectorizer = sentiment_vectorizers.get(aspect)

        if model is None or vectorizer is None:
            pred_labels = ["neutral"] * len(true_labels)
        elif isinstance(model, dict) and model.get("type") == "fallback":
            sentiment_names = {0: "positive", 1: "negative", 2: "neutral"}
            fallback = model.get("prediction", "neutral")
            if isinstance(fallback, int):
                fallback = sentiment_names.get(fallback, "neutral")
            pred_labels = [fallback] * len(true_labels)
        else:
            pipeline = PreprocessingPipeline()
            texts = pipeline.transform(subset["feedback"].tolist())
            X = vectorizer.transform(texts)
            pred_labels = model.predict(X).tolist()

        acc = float(accuracy_score(true_labels, pred_labels))
        macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))

        per_aspect[aspect] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "n_samples": len(true_labels),
        }

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    overall_acc = float(accuracy_score(all_true, all_pred)) if all_true else 0.0
    overall_macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else 0.0
    overall_weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0)) if all_true else 0.0

    per_aspect["_overall"] = {
        "accuracy": round(overall_acc, 4),
        "macro_f1": round(overall_macro_f1, 4),
        "weighted_f1": round(overall_weighted_f1, 4),
        "n_samples": len(all_true),
    }

    return per_aspect


def measure_inference_timing(model, tfidf, pipeline, texts) -> dict:
    """Measure inference time."""
    # Preprocess + TF-IDF
    processed = pipeline.transform(texts)
    X = tfidf.transform(processed)

    # Single sample
    start = time.time()
    for _ in range(100):
        model.predict(X[0:1])
    single_ms = (time.time() - start) * 1000 / 100

    # Full set
    start = time.time()
    model.predict(X)
    full_ms = (time.time() - start) * 1000

    return {
        "single_sample_ms": round(single_ms, 4),
        "full_test_set_ms": round(full_ms, 4),
        "per_sample_avg_ms": round(full_ms / len(texts), 4),
        "test_set_size": len(texts),
    }


def main():
    logger.info("=" * 70)
    logger.info("MODEL 1 EVALUATION — Generating model1_metrics.json")
    logger.info("=" * 70)

    # Config
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading data splits...")
    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load models
    logger.info("Loading Model 1 artifacts...")
    aspect_model = joblib.load(os.path.join(output_dir, "aspect_detector_lr.pkl"))
    mlb = joblib.load(os.path.join(output_dir, "mlb.pkl"))
    tfidf = TFIDFFeatures()
    tfidf.load(os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    pipeline = PreprocessingPipeline()
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.joblib")
    if os.path.exists(pipeline_path):
        pipeline.load(pipeline_path)

    sentiment_models = joblib.load(os.path.join(output_dir, "sentiment_classifiers_lr.pkl"))
    sentiment_vectorizers = joblib.load(os.path.join(output_dir, "sentiment_vectorizers_lr.pkl"))
    logger.info("All artifacts loaded")

    # Evaluate aspect detection
    logger.info("\nEvaluating aspect detection...")
    aspect_metrics = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        metrics = evaluate_aspect_detection(aspect_model, tfidf, pipeline, df, mlb, aspect_labels)
        aspect_metrics[split_name] = metrics
        logger.info(f"  {split_name}: Micro-F1={metrics['micro']['f1']}, Macro-F1={metrics['macro']['f1']}")

    # Evaluate sentiment
    logger.info("\nEvaluating sentiment classification...")
    sentiment_metrics = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        metrics = evaluate_sentiment(sentiment_models, sentiment_vectorizers, df, aspect_labels)
        sentiment_metrics[split_name] = metrics
        overall = metrics["_overall"]
        logger.info(f"  {split_name}: Macro-F1={overall['macro_f1']}, Weighted-F1={overall['weighted_f1']}")

    # Inference timing
    logger.info("\nMeasuring inference timing...")
    timing = measure_inference_timing(aspect_model, tfidf, pipeline, test_df["feedback"].tolist())
    logger.info(f"  Per-sample: {timing['per_sample_avg_ms']}ms")

    # Compile and save
    all_metrics = {
        "model": "Model 1 — Logistic Regression + TF-IDF",
        "aspect_detection": aspect_metrics,
        "sentiment_classification": sentiment_metrics,
        "inference_timing": timing,
    }

    metrics_path = os.path.join(output_dir, "model1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSaved: {metrics_path}")

    # Print summary
    print(f"\n{'═' * 60}")
    print("MODEL 1 METRICS SUMMARY")
    print(f"{'═' * 60}")
    print(f"\n  ASPECT DETECTION:")
    print(f"  {'Split':<8} {'Micro-F1':>10} {'Macro-F1':>10}")
    print(f"  {'─' * 30}")
    for split in ["train", "val", "test"]:
        m = aspect_metrics[split]
        print(f"  {split:<8} {m['micro']['f1']:>10.4f} {m['macro']['f1']:>10.4f}")

    print(f"\n  SENTIMENT CLASSIFICATION:")
    print(f"  {'Split':<8} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print(f"  {'─' * 32}")
    for split in ["train", "val", "test"]:
        o = sentiment_metrics[split]["_overall"]
        print(f"  {split:<8} {o['macro_f1']:>10.4f} {o['weighted_f1']:>12.4f}")

    print(f"\n  INFERENCE: {timing['per_sample_avg_ms']}ms/sample")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
