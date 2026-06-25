"""
Per-Aspect Sentiment Classifier Training for ABSA Telecom.

Trains a separate LogisticRegression sentiment model for each of the 15 aspects.
Each model classifies feedback into positive/negative/neutral for its specific aspect.
"""

import json
import logging
import os
import time
from collections import Counter

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 20  # Skip training if fewer samples


def load_split(path: str) -> pd.DataFrame:
    """Load CSV split and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    return df


def filter_by_aspect(df: pd.DataFrame, aspect: str) -> tuple:
    """
    Filter rows containing the given aspect and extract sentiment labels.

    Args:
        df: DataFrame with parsed aspects and aspect_sentiments columns
        aspect: Target aspect string

    Returns:
        (filtered_feedbacks: list[str], sentiment_labels: list[str])
    """
    mask = df["aspects"].apply(lambda aspects: aspect in aspects)
    filtered = df[mask].copy()

    feedbacks = filtered["feedback"].tolist()
    labels = filtered["aspect_sentiments"].apply(lambda d: d.get(aspect, "neutral")).tolist()

    return feedbacks, labels


def main():
    logger.info("=" * 70)
    logger.info("PER-ASPECT SENTIMENT CLASSIFIER TRAINING")
    logger.info("=" * 70)

    # ─── Config ───────────────────────────────────────────────────────────
    config = load_config()
    seed = config["seed"]
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    os.makedirs(output_dir, exist_ok=True)

    # ─── Load Splits ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("Loading data splits...")
    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ─── Train per-aspect models ──────────────────────────────────────────
    models = {}
    vectorizers = {}
    results_summary = []
    skipped_aspects = []

    total_start = time.time()

    for i, aspect in enumerate(aspect_labels, 1):
        logger.info("")
        logger.info(f"{'─' * 70}")
        logger.info(f"[{i:02d}/15] ASPECT: {aspect}")
        logger.info(f"{'─' * 70}")

        # Filter rows for this aspect
        train_texts, train_labels = filter_by_aspect(train_df, aspect)
        val_texts, val_labels = filter_by_aspect(val_df, aspect)
        test_texts, test_labels = filter_by_aspect(test_df, aspect)

        n_train = len(train_texts)
        train_dist = Counter(train_labels)

        logger.info(f"  Train samples: {n_train}")
        logger.info(f"  Class distribution: {dict(train_dist)}")
        logger.info(f"  Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

        # Check minimum sample threshold
        if n_train < MIN_TRAINING_SAMPLES:
            majority_class = train_dist.most_common(1)[0][0] if train_dist else "neutral"
            logger.warning(f"  ⚠️  SKIPPED: only {n_train} samples (< {MIN_TRAINING_SAMPLES}). "
                           f"Fallback: majority class = '{majority_class}'")
            models[aspect] = {"type": "fallback", "prediction": majority_class}
            vectorizers[aspect] = None
            skipped_aspects.append(aspect)
            results_summary.append({
                "aspect": aspect, "n_train": n_train, "status": "SKIPPED",
                "val_f1": None, "test_f1": None,
            })
            continue

        # Preprocessing
        pipeline = PreprocessingPipeline()
        train_processed = pipeline.fit_transform(train_texts)
        val_processed = pipeline.transform(val_texts)
        test_processed = pipeline.transform(test_texts)

        # TF-IDF (refit on each aspect's training subset)
        tfidf = TFIDFFeatures()
        X_train = tfidf.fit_transform(train_processed)
        X_val = tfidf.transform(val_processed)
        X_test = tfidf.transform(test_processed)

        logger.info(f"  TF-IDF shape: {X_train.shape}")

        # Train LogisticRegression
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )

        start = time.time()
        clf.fit(X_train, train_labels)
        train_time = time.time() - start

        # Evaluate
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)

        val_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        test_f1 = f1_score(test_labels, test_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_pred)
        test_acc = accuracy_score(test_labels, test_pred)

        logger.info(f"  Training time: {train_time:.2f}s")
        logger.info(f"  Val  — Macro-F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
        logger.info(f"  Test — Macro-F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}")

        # Store
        models[aspect] = clf
        vectorizers[aspect] = tfidf

        results_summary.append({
            "aspect": aspect, "n_train": n_train, "status": "TRAINED",
            "val_f1": round(val_f1, 4), "test_f1": round(test_f1, 4),
        })

    total_time = time.time() - total_start

    # ─── Save all models ──────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("─" * 70)

    models_path = os.path.join(output_dir, "sentiment_classifiers_lr.pkl")
    vectorizers_path = os.path.join(output_dir, "sentiment_vectorizers_lr.pkl")

    joblib.dump(models, models_path)
    logger.info(f"Models saved: {models_path}")

    joblib.dump(vectorizers, vectorizers_path)
    logger.info(f"Vectorizers saved: {vectorizers_path}")

    # ─── Summary Table ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    print(f"\n  {'Aspect':<28} {'N_train':>8} {'Status':<9} {'Val F1':>8} {'Test F1':>8}")
    print(f"  {'─' * 63}")

    for r in results_summary:
        val_str = f"{r['val_f1']:.4f}" if r['val_f1'] is not None else "  N/A"
        test_str = f"{r['test_f1']:.4f}" if r['test_f1'] is not None else "  N/A"
        print(f"  {r['aspect']:<28} {r['n_train']:>8} {r['status']:<9} {val_str:>8} {test_str:>8}")

    print(f"  {'─' * 63}")
    trained_count = sum(1 for r in results_summary if r["status"] == "TRAINED")
    avg_val_f1 = sum(r["val_f1"] for r in results_summary if r["val_f1"] is not None) / max(trained_count, 1)
    avg_test_f1 = sum(r["test_f1"] for r in results_summary if r["test_f1"] is not None) / max(trained_count, 1)

    print(f"\n  Models trained: {trained_count}/15")
    print(f"  Models skipped: {len(skipped_aspects)}")
    if skipped_aspects:
        print(f"  Skipped aspects: {skipped_aspects}")
    print(f"  Average Val Macro-F1:  {avg_val_f1:.4f}")
    print(f"  Average Test Macro-F1: {avg_test_f1:.4f}")
    print(f"  Total training time:   {total_time:.2f}s")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-ASPECT SENTIMENT TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
