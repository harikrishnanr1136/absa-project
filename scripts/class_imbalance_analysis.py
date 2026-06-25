"""
Class Imbalance Analysis for Multi-Label ABSA Telecom Dataset
Analyzes per-aspect sentiment distribution and multi-label imbalance.
"""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "absa_telecom_combined.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "eda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

VALID_ASPECTS = [
    "network_coverage", "internet_speed", "call_quality", "customer_support",
    "billing", "recharge_plans", "data_balance", "roaming", "sim_activation",
    "mobile_app_experience", "ott_bundle_services", "pricing",
    "value_for_money", "data_validity", "5g_experience",
]
SENTIMENTS = ["positive", "negative", "neutral"]
MIN_SAMPLE_THRESHOLD = 20


def load_data(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_aspect_sentiment_matrix(data: list) -> pd.DataFrame:
    """Build a DataFrame with counts: rows=aspects, cols=sentiments."""
    counts = {a: Counter() for a in VALID_ASPECTS}

    for entry in data:
        for aspect, sentiment in entry["aspect_sentiments"].items():
            if aspect in counts:
                counts[aspect][sentiment] += 1

    rows = []
    for aspect in VALID_ASPECTS:
        rows.append({
            "aspect": aspect,
            "positive": counts[aspect].get("positive", 0),
            "negative": counts[aspect].get("negative", 0),
            "neutral": counts[aspect].get("neutral", 0),
        })

    return pd.DataFrame(rows).set_index("aspect")


def plot_stacked_bar(matrix: pd.DataFrame):
    """1. Per-aspect sentiment distribution as stacked horizontal bars."""
    # Sort by total count descending
    matrix_sorted = matrix.copy()
    matrix_sorted["total"] = matrix_sorted.sum(axis=1)
    matrix_sorted = matrix_sorted.sort_values("total", ascending=True)
    matrix_sorted = matrix_sorted.drop(columns="total")

    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(12, 8))
    matrix_sorted.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Aspect", fontsize=12)
    ax.set_title("Per-Aspect Sentiment Distribution (Stacked)", fontsize=14, fontweight="bold")
    ax.legend(title="Sentiment", loc="lower right", fontsize=10)

    # Add total count labels
    for i, (idx, row) in enumerate(matrix_sorted.iterrows()):
        total = row.sum()
        ax.text(total + 2, i, str(int(total)), va="center", fontsize=9, color="gray")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "05_aspect_sentiment_stacked_bar.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_aspect_histogram(data: list):
    """4. Histogram of how many aspects appear per feedback entry."""
    aspect_counts = [len(entry["aspects"]) for entry in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    max_aspects = max(aspect_counts)
    bins = np.arange(0.5, max_aspects + 1.5, 1)

    counts, _, patches = ax.hist(aspect_counts, bins=bins, color="#3498db",
                                  edgecolor="white", linewidth=1.2, rwidth=0.8)

    # Add count labels on top of bars
    for patch, count in zip(patches, counts):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, count + 5,
                    str(int(count)), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Aspects per Entry", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Multi-Label Distribution: Aspects per Feedback Entry", fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, max_aspects + 1))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "06_aspects_per_entry_histogram.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def compute_summary_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute imbalance ratio and flag aspects needing attention."""
    summary_rows = []

    for aspect in matrix.index:
        pos = matrix.loc[aspect, "positive"]
        neg = matrix.loc[aspect, "negative"]
        neu = matrix.loc[aspect, "neutral"]

        counts = {"positive": pos, "negative": neg, "neutral": neu}
        max_sent = max(counts, key=counts.get)
        max_count = max(counts.values())
        min_count = min(counts.values())

        # Imbalance ratio (avoid division by zero)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        # Flag if any sentiment has fewer than threshold samples
        needs_attention = any(v < MIN_SAMPLE_THRESHOLD for v in counts.values())

        summary_rows.append({
            "aspect": aspect,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "total": pos + neg + neu,
            "dominant_sentiment": max_sent,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "needs_attention": needs_attention,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("total", ascending=False).reset_index(drop=True)
    return summary_df


def main():
    print("Loading dataset...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} entries.\n")

    # Build matrix
    matrix = build_aspect_sentiment_matrix(data)

    # Generate plots
    print("Generating visualizations:")
    plot_stacked_bar(matrix)
    plot_aspect_histogram(data)

    # Compute and print summary table
    summary = compute_summary_table(matrix)

    print(f"\n{'=' * 90}")
    print("CLASS IMBALANCE SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(f"\n{'Aspect':<28} {'Pos':>5} {'Neg':>5} {'Neu':>5} {'Total':>6} {'Dominant':<12} {'Ratio':>7} {'Attention'}")
    print(f"{'─' * 90}")

    for _, row in summary.iterrows():
        flag = "⚠️  YES" if row["needs_attention"] else "   no"
        print(f"{row['aspect']:<28} {row['positive']:>5} {row['negative']:>5} {row['neutral']:>5} "
              f"{row['total']:>6} {row['dominant_sentiment']:<12} {row['imbalance_ratio']:>7.2f} {flag}")

    print(f"{'─' * 90}")

    # Summary stats
    flagged = summary[summary["needs_attention"] == True]
    print(f"\n  Aspects needing attention (any sentiment < {MIN_SAMPLE_THRESHOLD}): {len(flagged)}/{len(summary)}")
    if len(flagged) > 0:
        print(f"  Flagged aspects: {', '.join(flagged['aspect'].tolist())}")

    avg_ratio = summary["imbalance_ratio"].replace(float("inf"), np.nan).mean()
    print(f"  Average imbalance ratio: {avg_ratio:.2f}")
    print(f"  Max imbalance ratio: {summary['imbalance_ratio'].max():.2f} ({summary.loc[summary['imbalance_ratio'].idxmax(), 'aspect']})")

    # Multi-label stats
    aspect_counts = [len(entry["aspects"]) for entry in data]
    print(f"\n  Multi-label stats:")
    print(f"    Average aspects/entry: {np.mean(aspect_counts):.2f}")
    print(f"    Single-aspect entries: {sum(1 for c in aspect_counts if c == 1)}")
    print(f"    Multi-aspect entries:  {sum(1 for c in aspect_counts if c >= 2)}")
    print(f"    Max aspects in entry:  {max(aspect_counts)}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
