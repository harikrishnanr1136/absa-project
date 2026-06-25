"""
Evaluation Script for DistilBERT ABSA Pipeline (Model 2).

Evaluates aspect detection and per-aspect sentiment classification on train/val/test.
Saves metrics, confusion matrices, and hardware report.
"""

import json
import logging
import os
import platform
import time
import tracemalloc
from multiprocessing import cpu_count

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, resolve_path
from src.evaluate import evaluate_model
from src.inference import ABSAInferencePipeline

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split(path: str) -> pd.DataFrame:
    """Load CSV and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    return df


def run_predictions(pipeline: ABSAInferencePipeline, texts: list) -> list:
    """Run pipeline predictions with timing."""
    start = time.time()
    results = pipeline.predict_batch(texts)
    elapsed = time.time() - start
    logger.info(f"  Predicted {len(texts)} samples in {elapsed:.2f}s ({elapsed*1000/len(texts):.2f}ms/sample)")
    return results


def evaluate_aspect_detection(predictions: list, df: pd.DataFrame, aspect_labels: list, mlb) -> dict:
    """
    Evaluate aspect detection: Hamming loss, micro/macro P/R/F1, per-aspect F1.
    """
    # Ground truth binary matrix
    Y_true = mlb.transform(df["aspects"])

    # Predicted binary matrix
    Y_pred = np.zeros((len(predictions), len(aspect_labels)), dtype=int)
    for i, pred in enumerate(predictions):
        for aspect in pred["detected_aspects"]:
            if aspect in aspect_labels:
                j = aspect_labels.index(aspect)
                Y_pred[i, j] = 1

    # Metrics
    h_loss = float(hamming_loss(Y_true, Y_pred))
    micro_p = float(precision_score(Y_true, Y_pred, average="micro", zero_division=0))
    micro_r = float(recall_score(Y_true, Y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    macro_p = float(precision_score(Y_true, Y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(Y_true, Y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))

    # Per-aspect F1
    per_aspect_f1 = {}
    for j, aspect in enumerate(aspect_labels):
        f1 = float(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0))
        per_aspect_f1[aspect] = round(f1, 4)

    return {
        "hamming_loss": round(h_loss, 4),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "per_aspect_f1": per_aspect_f1,
    }


def evaluate_sentiment_classification(predictions: list, df: pd.DataFrame, aspect_labels: list) -> dict:
    """
    Evaluate per-aspect sentiment: accuracy, macro F1, weighted F1, classification report.
    """
    results = {}
    all_true = []
    all_pred = []

    for aspect in aspect_labels:
        # Get ground truth for this aspect
        true_labels = []
        pred_labels = []

        for i, (_, row) in enumerate(df.iterrows()):
            if aspect in row["aspect_sentiments"]:
                true_sent = row["aspect_sentiments"][aspect]
                # Get prediction for this aspect (if detected)
                pred_sent = predictions[i]["aspect_sentiments"].get(aspect, "neutral")
                true_labels.append(true_sent)
                pred_labels.append(pred_sent)

        if not true_labels:
            results[aspect] = {"accuracy": 0, "macro_f1": 0, "weighted_f1": 0, "n_samples": 0}
            continue

        acc = float(accuracy_score(true_labels, pred_labels))
        macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))

        results[aspect] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "n_samples": len(true_labels),
        }

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    # Overall aggregated metrics
    overall_acc = float(accuracy_score(all_true, all_pred)) if all_true else 0.0
    overall_macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else 0.0
    overall_weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0)) if all_true else 0.0

    results["_overall"] = {
        "accuracy": round(overall_acc, 4),
        "macro_f1": round(overall_macro_f1, 4),
        "weighted_f1": round(overall_weighted_f1, 4),
        "n_samples": len(all_true),
    }

    return results


def plot_aspect_confusion_matrices(predictions: list, df: pd.DataFrame, aspect_labels: list, save_dir: str):
    """Plot and save binary confusion matrix per aspect for aspect detection."""
    mlb = MultiLabelBinarizer(classes=aspect_labels)
    mlb.fit([aspect_labels])
    Y_true = mlb.transform(df["aspects"])

    Y_pred = np.zeros((len(predictions), len(aspect_labels)), dtype=int)
    for i, pred in enumerate(predictions):
        for aspect in pred["detected_aspects"]:
            if aspect in aspect_labels:
                Y_pred[i, aspect_labels.index(aspect)] = 1

    # Plot 5x3 grid of confusion matrices
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for j, aspect in enumerate(aspect_labels):
        cm = confusion_matrix(Y_true[:, j], Y_pred[:, j])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[j],
                    xticklabels=["Not Present", "Present"],
                    yticklabels=["Not Present", "Present"])
        axes[j].set_title(aspect.replace("_", " ").title(), fontsize=10)
        axes[j].set_xlabel("Predicted")
        axes[j].set_ylabel("True")

    plt.suptitle("Aspect Detection — Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "distilbert_aspect_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"Aspect confusion matrices saved: {path}")


def plot_sentiment_confusion_matrices(predictions: list, df: pd.DataFrame, aspect_labels: list, save_dir: str):
    """Plot sentiment confusion matrix for top 6 aspects (most samples)."""
    # Find top 6 aspects by sample count on test
    aspect_counts = []
    for aspect in aspect_labels:
        count = sum(1 for _, row in df.iterrows() if aspect in row["aspect_sentiments"])
        aspect_counts.append((aspect, count))
    top_aspects = sorted(aspect_counts, key=lambda x: -x[1])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    labels_order = ["positive", "negative", "neutral"]

    for idx, (aspect, _) in enumerate(top_aspects):
        true_labels = []
        pred_labels = []

        for i, (_, row) in enumerate(df.iterrows()):
            if aspect in row["aspect_sentiments"]:
                true_labels.append(row["aspect_sentiments"][aspect])
                pred_labels.append(predictions[i]["aspect_sentiments"].get(aspect, "neutral"))

        if not true_labels:
            continue

        cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes[idx],
                    xticklabels=labels_order, yticklabels=labels_order)
        axes[idx].set_title(aspect.replace("_", " ").title(), fontsize=11)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.suptitle("Sentiment Classification — Confusion Matrices (Test Set, Top 6 Aspects)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "distilbert_sentiment_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"Sentiment confusion matrices saved: {path}")


def measure_inference_timing(pipeline: ABSAInferencePipeline, texts: list) -> dict:
    """Measure inference timing: single sample, full set, per-sample average."""
    # Single sample (average over 10 runs)
    single_text = texts[0]
    start = time.time()
    for _ in range(10):
        pipeline.predict(single_text)
    single_ms = (time.time() - start) * 1000 / 10

    # Full test set
    start = time.time()
    pipeline.predict_batch(texts)
    full_ms = (time.time() - start) * 1000

    per_sample_ms = full_ms / len(texts)

    return {
        "single_sample_ms": round(single_ms, 4),
        "full_test_set_ms": round(full_ms, 4),
        "per_sample_avg_ms": round(per_sample_ms, 4),
        "test_set_size": len(texts),
    }


def get_hardware_info() -> dict:
    """Collect hardware info."""
    gpu_available = torch.cuda.is_available()
    gpu_memory = torch.cuda.memory_allocated() / (1024**2) if gpu_available else 0.0

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count(),
        "gpu_available": gpu_available,
        "gpu_memory_mb": round(gpu_memory, 2),
        "torch_version": torch.__version__,
    }


def main():
    logger.info("=" * 70)
    logger.info("DISTILBERT ABSA EVALUATION (Model 2)")
    logger.info("=" * 70)

    # ─── Config & Paths ───────────────────────────────────────────────────
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)

    # ─── Load Data ────────────────────────────────────────────────────────
    logger.info("Loading data splits...")
    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ─── Initialize Pipeline ──────────────────────────────────────────────
    logger.info("Initializing ABSAInferencePipeline...")
    pipeline = ABSAInferencePipeline()

    # MLB for aspect evaluation
    mlb = MultiLabelBinarizer(classes=aspect_labels)
    mlb.fit([aspect_labels])

    # ─── Run Predictions ──────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("Running predictions on all splits...")
    logger.info("─" * 70)

    tracemalloc.start()

    logger.info("\n  Train set:")
    train_preds = run_predictions(pipeline, train_df["feedback"].tolist())
    logger.info("\n  Val set:")
    val_preds = run_predictions(pipeline, val_df["feedback"].tolist())
    logger.info("\n  Test set:")
    test_preds = run_predictions(pipeline, test_df["feedback"].tolist())

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ─── Aspect Detection Evaluation ──────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("ASPECT DETECTION EVALUATION")
    logger.info("─" * 70)

    aspect_metrics = {}
    for split_name, preds, df in [("train", train_preds, train_df),
                                    ("val", val_preds, val_df),
                                    ("test", test_preds, test_df)]:
        metrics = evaluate_aspect_detection(preds, df, aspect_labels, mlb)
        aspect_metrics[split_name] = metrics
        logger.info(f"\n  {split_name.upper()}:")
        logger.info(f"    Hamming Loss: {metrics['hamming_loss']}")
        logger.info(f"    Micro F1: {metrics['micro']['f1']} | Macro F1: {metrics['macro']['f1']}")

    # ─── Sentiment Classification Evaluation ──────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("SENTIMENT CLASSIFICATION EVALUATION")
    logger.info("─" * 70)

    sentiment_metrics = {}
    for split_name, preds, df in [("train", train_preds, train_df),
                                    ("val", val_preds, val_df),
                                    ("test", test_preds, test_df)]:
        metrics = evaluate_sentiment_classification(preds, df, aspect_labels)
        sentiment_metrics[split_name] = metrics
        overall = metrics["_overall"]
        logger.info(f"\n  {split_name.upper()} (overall):")
        logger.info(f"    Accuracy: {overall['accuracy']} | Macro F1: {overall['macro_f1']} | "
                    f"Weighted F1: {overall['weighted_f1']}")

    # ─── Timing ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("INFERENCE TIMING")
    logger.info("─" * 70)

    timing = measure_inference_timing(pipeline, test_df["feedback"].tolist())
    logger.info(f"  Single sample: {timing['single_sample_ms']:.2f}ms")
    logger.info(f"  Full test set: {timing['full_test_set_ms']:.2f}ms")
    logger.info(f"  Per-sample avg: {timing['per_sample_avg_ms']:.2f}ms")

    # ─── Confusion Matrix Plots ───────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("GENERATING CONFUSION MATRIX PLOTS")
    logger.info("─" * 70)

    plot_aspect_confusion_matrices(test_preds, test_df, aspect_labels, eda_dir)
    plot_sentiment_confusion_matrices(test_preds, test_df, aspect_labels, eda_dir)

    # ─── Hardware Report ──────────────────────────────────────────────────
    hardware = get_hardware_info()
    hardware["peak_inference_memory_mb"] = round(peak_memory / (1024**2), 3)

    # ─── Save Metrics ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("SAVING RESULTS")
    logger.info("─" * 70)

    all_metrics = {
        "model": "Model 2 — DistilBERT ABSA",
        "aspect_detection": aspect_metrics,
        "sentiment_classification": sentiment_metrics,
        "inference_timing": timing,
    }

    metrics_path = os.path.join(output_dir, "model2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    hw_path = os.path.join(output_dir, "model2_hardware_report.json")
    with open(hw_path, "w") as f:
        json.dump(hardware, f, indent=2)
    logger.info(f"Hardware report saved: {hw_path}")

    # ─── Print Summary ────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("MODEL 2 EVALUATION SUMMARY")
    print(f"{'═' * 70}")

    print(f"\n{'─' * 70}")
    print("ASPECT DETECTION:")
    print(f"{'─' * 70}")
    print(f"  {'Split':<8} {'Hamming':>9} {'Micro-F1':>10} {'Macro-F1':>10}")
    print(f"  {'─' * 40}")
    for split in ["train", "val", "test"]:
        m = aspect_metrics[split]
        print(f"  {split:<8} {m['hamming_loss']:>9.4f} {m['micro']['f1']:>10.4f} {m['macro']['f1']:>10.4f}")

    print(f"\n{'─' * 70}")
    print("SENTIMENT CLASSIFICATION:")
    print(f"{'─' * 70}")
    print(f"  {'Split':<8} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print(f"  {'─' * 42}")
    for split in ["train", "val", "test"]:
        o = sentiment_metrics[split]["_overall"]
        print(f"  {split:<8} {o['accuracy']:>10.4f} {o['macro_f1']:>10.4f} {o['weighted_f1']:>12.4f}")

    print(f"\n{'─' * 70}")
    print("INFERENCE:")
    print(f"{'─' * 70}")
    print(f"  Single sample:  {timing['single_sample_ms']:.2f} ms")
    print(f"  Per-sample avg: {timing['per_sample_avg_ms']:.2f} ms")
    print(f"  Peak memory:    {hardware['peak_inference_memory_mb']:.2f} MB")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
