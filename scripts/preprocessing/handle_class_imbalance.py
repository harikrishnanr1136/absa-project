"""
Class Imbalance Analysis and Treatment for Telecom ABSA Dataset

This script:
1. Analyzes aspect detection imbalance (multi-label frequency)
2. Analyzes sentiment imbalance per aspect
3. Applies oversampling (RandomOverSampler) for aspects where any sentiment < 15 samples
4. Plots before/after comparison for the top 3 most imbalanced aspects

NOTE on class_weight="balanced" for aspect detection:
    In a multi-label classification setting (aspect detection), class_weight="balanced"
    adjusts the loss function to give higher weight to underrepresented aspect classes.
    The formula is: weight_j = n_samples / (n_classes * n_samples_with_aspect_j)
    This is applied at MODEL TRAINING time (e.g., in sklearn classifiers or PyTorch
    loss functions like BCEWithLogitsLoss with pos_weight), not during preprocessing.
    We flag underrepresented aspects here so the training script can apply appropriate
    weighting.
"""

import json
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "absa_telecom_combined.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "eda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ASPECT_FREQUENCY_THRESHOLD = 50  # Aspects with fewer samples are flagged
SENTIMENT_MIN_THRESHOLD = 15     # Minimum samples per sentiment per aspect

VALID_ASPECTS = [
    "network_coverage", "internet_speed", "call_quality", "customer_support",
    "billing", "recharge_plans", "data_balance", "roaming", "sim_activation",
    "mobile_app_experience", "ott_bundle_services", "pricing",
    "value_for_money", "data_validity", "5g_experience",
]
SENTIMENTS = ["positive", "negative", "neutral"]

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    logger.info(f"Loaded {len(df)} entries from {path}")
    return df


def analyze_aspect_frequency(df: pd.DataFrame) -> dict:
    """
    Compute aspect frequency and flag underrepresented aspects.

    For multi-label aspect detection, class_weight="balanced" should be applied
    at model training time to compensate for frequency imbalance. Here we identify
    which aspects need attention.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("ASPECT DETECTION IMBALANCE (Multi-Label Frequency)")
    logger.info("=" * 60)

    aspect_freq = Counter()
    for aspects_list in df["aspects"]:
        for a in aspects_list:
            aspect_freq[a] += 1

    logger.info(f"\n  {'Aspect':<28} {'Count':>6} {'Status'}")
    logger.info(f"  {'─' * 50}")

    underrepresented = []
    for aspect in sorted(VALID_ASPECTS, key=lambda x: aspect_freq.get(x, 0), reverse=True):
        count = aspect_freq.get(aspect, 0)
        status = "⚠️  UNDERREPRESENTED" if count < ASPECT_FREQUENCY_THRESHOLD else "OK"
        if count < ASPECT_FREQUENCY_THRESHOLD:
            underrepresented.append(aspect)
        logger.info(f"  {aspect:<28} {count:>6} {status}")

    logger.info(f"  {'─' * 50}")
    logger.info(f"  Threshold: {ASPECT_FREQUENCY_THRESHOLD} samples")
    logger.info(f"  Underrepresented aspects: {len(underrepresented)}")
    if underrepresented:
        logger.info(f"  Flagged: {underrepresented}")
        # NOTE: At model training time, apply class_weight="balanced" or compute
        # pos_weight for BCEWithLogitsLoss to handle these imbalanced aspects.
        # Formula: pos_weight_j = (total - positive_j) / positive_j
        logger.info("  ACTION: Apply class_weight='balanced' during model training")

    return aspect_freq


def analyze_sentiment_per_aspect(df: pd.DataFrame) -> dict:
    """
    For each aspect, compute sentiment distribution and identify imbalanced aspects.

    Returns dict: {aspect: {"positive": count, "negative": count, "neutral": count}}
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("SENTIMENT IMBALANCE PER ASPECT")
    logger.info("=" * 60)

    aspect_sentiment_counts = {a: Counter() for a in VALID_ASPECTS}

    for _, row in df.iterrows():
        for aspect, sentiment in row["aspect_sentiments"].items():
            if aspect in aspect_sentiment_counts:
                aspect_sentiment_counts[aspect][sentiment] += 1

    logger.info(f"\n  {'Aspect':<28} {'Pos':>5} {'Neg':>5} {'Neu':>5} {'Min':>5} {'Imbalanced?'}")
    logger.info(f"  {'─' * 65}")

    imbalanced_aspects = {}
    for aspect in VALID_ASPECTS:
        counts = aspect_sentiment_counts[aspect]
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0)
        min_count = min(pos, neg, neu)
        is_imbalanced = min_count < SENTIMENT_MIN_THRESHOLD

        status = f"⚠️  YES (min={min_count})" if is_imbalanced else "NO"
        logger.info(f"  {aspect:<28} {pos:>5} {neg:>5} {neu:>5} {min_count:>5} {status}")

        if is_imbalanced:
            imbalanced_aspects[aspect] = {"positive": pos, "negative": neg, "neutral": neu}

    logger.info(f"  {'─' * 65}")
    logger.info(f"  Threshold: {SENTIMENT_MIN_THRESHOLD} samples per sentiment")
    logger.info(f"  Aspects needing oversampling: {len(imbalanced_aspects)}")

    return aspect_sentiment_counts, imbalanced_aspects


def oversample_aspect(df: pd.DataFrame, aspect: str, before_counts: dict) -> tuple:
    """
    Apply RandomOverSampler to a specific aspect's data.

    Extracts rows containing the aspect, uses feedback index as feature proxy,
    and oversamples the minority sentiment class.

    Returns:
        (after_counts dict, num_samples_added)
    """
    # Extract rows that contain this aspect
    mask = df["aspects"].apply(lambda x: aspect in x)
    aspect_df = df[mask].copy()

    # Get sentiment labels for this aspect
    aspect_df["_sentiment"] = aspect_df["aspect_sentiments"].apply(
        lambda x: x.get(aspect, "neutral")
    )

    # Create a dummy feature matrix (row indices as features for oversampling)
    X = aspect_df.index.values.reshape(-1, 1)
    y = aspect_df["_sentiment"].values

    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Compute after counts
    after_counts = Counter(y_resampled)
    samples_added = len(y_resampled) - len(y)

    return dict(after_counts), samples_added


def apply_oversampling(df: pd.DataFrame, aspect_sentiment_counts: dict,
                       imbalanced_aspects: dict) -> dict:
    """
    Apply oversampling to all imbalanced aspects and log results.

    Returns dict of {aspect: {"before": counts, "after": counts}}
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("OVERSAMPLING TREATMENT")
    logger.info("=" * 60)

    results = {}

    for aspect, before_counts in imbalanced_aspects.items():
        after_counts, samples_added = oversample_aspect(df, aspect, before_counts)

        results[aspect] = {
            "before": before_counts,
            "after": after_counts,
        }

        logger.info(f"\n  Aspect: {aspect}")
        logger.info(f"    Before: pos={before_counts['positive']}, "
                    f"neg={before_counts['negative']}, "
                    f"neu={before_counts['neutral']}")
        logger.info(f"    After:  pos={after_counts.get('positive', 0)}, "
                    f"neg={after_counts.get('negative', 0)}, "
                    f"neu={after_counts.get('neutral', 0)}")
        logger.info(f"    Samples added: +{samples_added}")

    logger.info(f"\n  Total aspects oversampled: {len(results)}")
    return results


def plot_imbalance_before_after(results: dict):
    """
    Plot side-by-side bar charts of sentiment distribution before and after
    oversampling for the top 3 most imbalanced aspects.
    """
    if not results:
        logger.info("  No imbalanced aspects to plot.")
        return

    # Rank by imbalance ratio (max/min before oversampling)
    def imbalance_ratio(counts):
        values = [counts.get(s, 0) for s in SENTIMENTS]
        min_val = min(values) if min(values) > 0 else 1
        return max(values) / min_val

    ranked = sorted(results.keys(), key=lambda a: imbalance_ratio(results[a]["before"]),
                    reverse=True)
    top_3 = ranked[:3]

    fig, axes = plt.subplots(1, len(top_3), figsize=(6 * len(top_3), 5))
    if len(top_3) == 1:
        axes = [axes]

    colors_before = ["#a8d5a2", "#f5a3a3", "#c8c8c8"]
    colors_after = ["#2ecc71", "#e74c3c", "#95a5a6"]

    for ax, aspect in zip(axes, top_3):
        before = results[aspect]["before"]
        after = results[aspect]["after"]

        x = np.arange(len(SENTIMENTS))
        width = 0.35

        before_vals = [before.get(s, 0) for s in SENTIMENTS]
        after_vals = [after.get(s, 0) for s in SENTIMENTS]

        bars1 = ax.bar(x - width / 2, before_vals, width, label="Before",
                       color=colors_before, edgecolor="gray", linewidth=0.8)
        bars2 = ax.bar(x + width / 2, after_vals, width, label="After",
                       color=colors_after, edgecolor="gray", linewidth=0.8)

        # Add count labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha="center", fontsize=9, color="gray")
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(SENTIMENTS, fontsize=10)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(aspect.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("Sentiment Distribution: Before vs After Oversampling",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "imbalance_treatment.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"\n  Plot saved: {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("CLASS IMBALANCE ANALYSIS & TREATMENT")
    logger.info("Telecom ABSA Dataset")
    logger.info("=" * 60)

    # Load dataset
    df = load_dataset(INPUT_PATH)

    # Step 1: Analyze aspect frequency (multi-label imbalance)
    aspect_freq = analyze_aspect_frequency(df)

    # Step 2: Analyze sentiment imbalance per aspect
    aspect_sentiment_counts, imbalanced_aspects = analyze_sentiment_per_aspect(df)

    # Step 3: Apply oversampling to imbalanced aspects
    results = apply_oversampling(df, aspect_sentiment_counts, imbalanced_aspects)

    # Step 4: Plot before/after for top 3 most imbalanced
    plot_imbalance_before_after(results)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
