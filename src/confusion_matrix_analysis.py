"""
Confusion Matrix Analysis for Both Models.

Generates normalized confusion matrices for sentiment classification
across all 15 aspects, identifies most confused aspects, and creates
side-by-side comparison plots.
"""

import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["positive", "negative", "neutral"]


def load_test_data(config: dict) -> pd.DataFrame:
    """Load and parse test.csv."""
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    logger.info(f"Loaded test set: {len(df)} rows")
    return df


def get_model1_predictions(test_df: pd.DataFrame, config: dict) -> dict:
    """
    Get Model 1 (LR) per-aspect sentiment predictions on test set.
    Returns {aspect: [pred_labels]} for each aspect.
    """
    output_dir = resolve_path(config["outputs"]["models"])
    aspect_labels = config["labels"]["aspects"]

    sentiment_models = joblib.load(os.path.join(output_dir, "sentiment_classifiers_lr.pkl"))
    sentiment_vectorizers = joblib.load(os.path.join(output_dir, "sentiment_vectorizers_lr.pkl"))

    predictions = {}

    for aspect in aspect_labels:
        mask = test_df["aspects"].apply(lambda a: aspect in a)
        subset = test_df[mask]
        if len(subset) == 0:
            continue

        model = sentiment_models.get(aspect)
        vectorizer = sentiment_vectorizers.get(aspect)

        if model is None or vectorizer is None:
            predictions[aspect] = ["neutral"] * len(subset)
            continue

        if isinstance(model, dict) and model.get("type") == "fallback":
            sentiment_names = {0: "positive", 1: "negative", 2: "neutral"}
            fallback = model.get("prediction", "neutral")
            if isinstance(fallback, int):
                fallback = sentiment_names.get(fallback, "neutral")
            predictions[aspect] = [fallback] * len(subset)
        else:
            pipeline = PreprocessingPipeline()
            texts = pipeline.transform(subset["feedback"].tolist())
            X = vectorizer.transform(texts)
            predictions[aspect] = model.predict(X).tolist()

    logger.info(f"Model 1 predictions generated for {len(predictions)} aspects")
    return predictions


def get_model2_predictions(test_df: pd.DataFrame, config: dict) -> dict:
    """
    Get Model 2 (DistilBERT) per-aspect sentiment predictions on test set.
    Uses ABSAInferencePipeline.
    Returns {aspect: [pred_labels]}.
    """
    try:
        from src.inference import ABSAInferencePipeline
        pipeline = ABSAInferencePipeline()
    except Exception as e:
        logger.warning(f"Could not load Model 2 pipeline: {e}")
        return {}

    aspect_labels = config["labels"]["aspects"]
    feedbacks = test_df["feedback"].tolist()

    logger.info(f"Running Model 2 predictions on {len(feedbacks)} test samples...")
    batch_results = pipeline.predict_batch(feedbacks)

    predictions = {}
    for aspect in aspect_labels:
        mask = test_df["aspects"].apply(lambda a: aspect in a)
        indices = mask[mask].index.tolist()
        if not indices:
            continue

        preds = []
        for idx in indices:
            result = batch_results[idx]
            pred_sent = result.get("aspect_sentiments", {}).get(aspect, "neutral")
            preds.append(pred_sent)

        predictions[aspect] = preds

    logger.info(f"Model 2 predictions generated for {len(predictions)} aspects")
    return predictions


def get_true_labels(test_df: pd.DataFrame, aspect: str) -> list:
    """Get ground truth sentiment labels for a specific aspect."""
    mask = test_df["aspects"].apply(lambda a: aspect in a)
    subset = test_df[mask]
    return subset["aspect_sentiments"].apply(lambda d: d.get(aspect, "neutral")).tolist()


def compute_confusion_data(true_labels: list, pred_labels: list) -> dict:
    """Compute confusion matrix and off-diagonal confusion score."""
    cm = confusion_matrix(true_labels, pred_labels, labels=SENTIMENT_LABELS)

    # Normalize by true labels
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm_norm / row_sums

    # Off-diagonal score (higher = more confused)
    off_diag = cm_norm.copy()
    np.fill_diagonal(off_diag, 0)
    confusion_score = off_diag.sum()

    # Find most confused pair
    max_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
    most_confused_from = SENTIMENT_LABELS[max_idx[0]]
    most_confused_to = SENTIMENT_LABELS[max_idx[1]]
    most_confused_pct = float(off_diag[max_idx])

    acc = accuracy_score(true_labels, pred_labels)

    return {
        "cm": cm.tolist(),
        "cm_normalized": cm_norm.tolist(),
        "accuracy": round(acc, 4),
        "confusion_score": round(confusion_score, 4),
        "most_confused": {
            "from": most_confused_from,
            "to": most_confused_to,
            "percentage": round(most_confused_pct * 100, 1),
        },
    }


def plot_all_aspects_grid(confusion_data: dict, model_name: str, save_path: str):
    """Plot 5x3 grid of confusion matrices for all aspects."""
    aspects = sorted(confusion_data.keys())

    fig, axes = plt.subplots(5, 3, figsize=(15, 22))
    axes = axes.flatten()

    for i, aspect in enumerate(aspects):
        if i >= 15:
            break
        ax = axes[i]
        data = confusion_data[aspect]
        cm_norm = np.array(data["cm_normalized"])
        acc = data["accuracy"]

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS,
                    ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(f"{aspect.replace('_', ' ').title()}\n(acc={acc:.2f})", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused axes
    for i in range(len(aspects), 15):
        axes[i].set_visible(False)

    plt.suptitle(f"Sentiment Confusion Matrices — {model_name} (Normalized by True Label)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_top3_comparison(m1_data: dict, m2_data: dict, top3_aspects: list, save_path: str):
    """Plot side-by-side confusion matrices for top 3 confused aspects."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    models = [("Model 1 (LR)", m1_data), ("Model 2 (DistilBERT)", m2_data)]

    for row, (model_name, data) in enumerate(models):
        for col, aspect in enumerate(top3_aspects):
            ax = axes[row, col]
            if aspect in data:
                cm_norm = np.array(data[aspect]["cm_normalized"])
                acc = data[aspect]["accuracy"]
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="RdYlGn",
                            xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS,
                            ax=ax, vmin=0, vmax=1, cbar=False)
                title = f"{model_name}\n{aspect.replace('_', ' ').title()} (acc={acc:.2f})"
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                title = f"{model_name}\n{aspect.replace('_', ' ').title()}"

            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel("True")
            if row == 1:
                ax.set_xlabel("Predicted")

    plt.suptitle("Top 3 Most Confused Aspects — Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    logger.info("=" * 70)
    logger.info("CONFUSION MATRIX ANALYSIS")
    logger.info("=" * 70)

    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(eda_dir, exist_ok=True)

    # Load test data
    test_df = load_test_data(config)

    # Get predictions from both models
    logger.info("\nGenerating Model 1 predictions...")
    m1_preds = get_model1_predictions(test_df, config)

    logger.info("\nGenerating Model 2 predictions...")
    m2_preds = get_model2_predictions(test_df, config)

    # Compute confusion matrices for all aspects
    logger.info("\nComputing confusion matrices...")
    m1_confusion = {}
    m2_confusion = {}

    for aspect in aspect_labels:
        true_labels = get_true_labels(test_df, aspect)
        if not true_labels:
            continue

        if aspect in m1_preds and len(m1_preds[aspect]) == len(true_labels):
            m1_confusion[aspect] = compute_confusion_data(true_labels, m1_preds[aspect])

        if aspect in m2_preds and len(m2_preds[aspect]) == len(true_labels):
            m2_confusion[aspect] = compute_confusion_data(true_labels, m2_preds[aspect])

    # Plot all-aspect grids
    logger.info("\nGenerating plots...")
    if m1_confusion:
        plot_all_aspects_grid(m1_confusion, "Model 1 (LR+TF-IDF)",
                              os.path.join(eda_dir, "model1_confusion_matrices_all_aspects.png"))

    if m2_confusion:
        plot_all_aspects_grid(m2_confusion, "Model 2 (DistilBERT)",
                              os.path.join(eda_dir, "model2_confusion_matrices_all_aspects.png"))

    # Identify top 3 most confused aspects per model
    logger.info("\n" + "─" * 70)
    logger.info("TOP 3 MOST CONFUSED ASPECTS")
    logger.info("─" * 70)

    def get_top_confused(confusion_data, model_name):
        sorted_aspects = sorted(confusion_data.items(),
                                key=lambda x: x[1]["confusion_score"], reverse=True)
        top3 = sorted_aspects[:3]

        print(f"\n  {model_name}:")
        for aspect, data in top3:
            mc = data["most_confused"]
            print(f"    • {aspect.replace('_', ' ').title()}: "
                  f"'{mc['from']}' misclassified as '{mc['to']}' ({mc['percentage']:.1f}%)")

        return [a for a, _ in top3]

    m1_top3 = get_top_confused(m1_confusion, "Model 1") if m1_confusion else []
    m2_top3 = get_top_confused(m2_confusion, "Model 2") if m2_confusion else []

    # Use union of top3 for comparison plot
    combined_top3 = list(dict.fromkeys(m1_top3 + m2_top3))[:3]

    if combined_top3 and m1_confusion and m2_confusion:
        plot_top3_comparison(m1_confusion, m2_confusion, combined_top3,
                            os.path.join(eda_dir, "top3_confused_aspects_comparison.png"))

    # Find example misclassified feedbacks
    logger.info("\n" + "─" * 70)
    logger.info("EXAMPLE MISCLASSIFIED FEEDBACKS")
    logger.info("─" * 70)

    misclassified_examples = {}
    for aspect in combined_top3[:3]:
        true_labels = get_true_labels(test_df, aspect)
        pred_labels = m1_preds.get(aspect, [])

        if len(true_labels) != len(pred_labels):
            continue

        mask = test_df["aspects"].apply(lambda a: aspect in a)
        subset = test_df[mask].reset_index(drop=True)

        examples = []
        for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
            if t != p and i < len(subset):
                examples.append({
                    "feedback": subset.iloc[i]["feedback"][:100],
                    "true": t,
                    "predicted": p,
                })
            if len(examples) >= 3:
                break

        if examples:
            misclassified_examples[aspect] = examples
            print(f"\n  {aspect.replace('_', ' ').title()}:")
            for ex in examples:
                print(f"    True={ex['true']}, Pred={ex['predicted']}")
                print(f"    \"{ex['feedback']}\"")

    # Save all data to JSON
    output_data = {
        "model1_confusion": {k: {kk: vv for kk, vv in v.items() if kk != "cm_normalized"}
                             for k, v in m1_confusion.items()},
        "model2_confusion": {k: {kk: vv for kk, vv in v.items() if kk != "cm_normalized"}
                             for k, v in m2_confusion.items()},
        "model1_top3_confused": m1_top3,
        "model2_top3_confused": m2_top3,
        "misclassified_examples": misclassified_examples,
    }

    json_path = os.path.join(output_dir, "confusion_matrix_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_path}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
