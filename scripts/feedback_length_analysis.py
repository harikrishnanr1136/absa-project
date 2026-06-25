"""
Feedback Length Distribution Analysis for ABSA Telecom Dataset
Analyzes how feedback length relates to source channel, aspect count, and sentiment.
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


def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # Derived columns
    df["feedback_length"] = df["feedback"].apply(lambda x: len(x.split()))
    df["aspect_count"] = df["aspects"].apply(len)
    df["aspect_group"] = df["aspect_count"].apply(
        lambda x: "1 aspect" if x == 1 else ("2 aspects" if x == 2 else "3+ aspects")
    )

    # Dominant sentiment: most frequent sentiment in aspect_sentiments
    def get_dominant_sentiment(sent_dict):
        if not sent_dict:
            return "neutral"
        counts = Counter(sent_dict.values())
        return counts.most_common(1)[0][0]

    df["dominant_sentiment"] = df["aspect_sentiments"].apply(get_dominant_sentiment)

    return df


def plot_length_histogram(df: pd.DataFrame):
    """1. Histogram of feedback_length with KDE overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["feedback_length"], bins=30, kde=True, color="#3498db",
                 edgecolor="white", linewidth=0.8, ax=ax)

    mean_len = df["feedback_length"].mean()
    median_len = df["feedback_length"].median()
    ax.axvline(mean_len, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_len:.1f}")
    ax.axvline(median_len, color="#2ecc71", linestyle="-.", linewidth=1.5, label=f"Median: {median_len:.1f}")

    ax.set_xlabel("Feedback Length (words)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Feedback Length Distribution with KDE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "07_feedback_length_histogram.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_boxplot_by_channel(df: pd.DataFrame):
    """2. Box plots of feedback_length grouped by source_channel."""
    # Order channels by median length
    channel_order = (
        df.groupby("source_channel")["feedback_length"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="source_channel", y="feedback_length",
        order=channel_order, palette="Set2", ax=ax,
        linewidth=1.2, fliersize=3
    )

    ax.set_xlabel("Source Channel", fontsize=12)
    ax.set_ylabel("Feedback Length (words)", fontsize=12)
    ax.set_title("Feedback Length by Source Channel", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "08_length_boxplot_by_channel.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_boxplot_by_aspect_count(df: pd.DataFrame):
    """3. Box plots of feedback_length grouped by aspect count."""
    group_order = ["1 aspect", "2 aspects", "3+ aspects"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="aspect_group", y="feedback_length",
        order=group_order, palette="coolwarm", ax=ax,
        linewidth=1.2, fliersize=3
    )

    ax.set_xlabel("Number of Aspects", fontsize=12)
    ax.set_ylabel("Feedback Length (words)", fontsize=12)
    ax.set_title("Feedback Length by Aspect Count", fontsize=14, fontweight="bold")

    # Add mean markers
    means = df.groupby("aspect_group")["feedback_length"].mean()
    for i, group in enumerate(group_order):
        ax.scatter(i, means[group], color="red", s=80, zorder=5, marker="D",
                   label="Mean" if i == 0 else "")
    ax.legend(fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "09_length_boxplot_by_aspects.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_scatter_length_vs_aspects(df: pd.DataFrame):
    """4. Scatter plot: feedback_length vs aspect count, colored by dominant sentiment."""
    color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for sentiment, color in color_map.items():
        subset = df[df["dominant_sentiment"] == sentiment]
        ax.scatter(
            subset["aspect_count"] + np.random.uniform(-0.15, 0.15, size=len(subset)),
            subset["feedback_length"],
            c=color, alpha=0.5, s=25, label=sentiment, edgecolors="none"
        )

    ax.set_xlabel("Number of Aspects", fontsize=12)
    ax.set_ylabel("Feedback Length (words)", fontsize=12)
    ax.set_title("Feedback Length vs Aspect Count (colored by Dominant Sentiment)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Dominant Sentiment", fontsize=10, markerscale=2)
    ax.set_xticks(range(1, df["aspect_count"].max() + 1))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "10_scatter_length_vs_aspects.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def print_summary_stats(df: pd.DataFrame):
    """5. Print mean length per source_channel and per aspect count group."""
    print(f"\n{'=' * 60}")
    print("FEEDBACK LENGTH SUMMARY STATISTICS")
    print(f"{'=' * 60}")

    print(f"\n{'─' * 60}")
    print("Mean Feedback Length by Source Channel")
    print(f"{'─' * 60}")
    channel_stats = (
        df.groupby("source_channel")["feedback_length"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(f"\n  {'Channel':<22} {'Mean':>7} {'Median':>8} {'Std':>7} {'Count':>7}")
    print(f"  {'─' * 53}")
    for ch, row in channel_stats.iterrows():
        print(f"  {ch:<22} {row['mean']:>7.1f} {row['median']:>8.1f} {row['std']:>7.1f} {int(row['count']):>7}")

    print(f"\n{'─' * 60}")
    print("Mean Feedback Length by Aspect Count Group")
    print(f"{'─' * 60}")
    group_order = ["1 aspect", "2 aspects", "3+ aspects"]
    group_stats = (
        df.groupby("aspect_group")["feedback_length"]
        .agg(["mean", "median", "std", "count"])
        .reindex(group_order)
    )
    print(f"\n  {'Group':<15} {'Mean':>7} {'Median':>8} {'Std':>7} {'Count':>7}")
    print(f"  {'─' * 46}")
    for grp, row in group_stats.iterrows():
        print(f"  {grp:<15} {row['mean']:>7.1f} {row['median']:>8.1f} {row['std']:>7.1f} {int(row['count']):>7}")

    print(f"\n{'─' * 60}")
    print("Overall Length Stats")
    print(f"{'─' * 60}")
    print(f"  Mean:   {df['feedback_length'].mean():.1f} words")
    print(f"  Median: {df['feedback_length'].median():.1f} words")
    print(f"  Std:    {df['feedback_length'].std():.1f} words")
    print(f"  Min:    {df['feedback_length'].min()} words")
    print(f"  Max:    {df['feedback_length'].max()} words")
    print(f"{'=' * 60}\n")


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} entries.\n")

    print("Generating visualizations:")
    plot_length_histogram(df)
    plot_boxplot_by_channel(df)
    plot_boxplot_by_aspect_count(df)
    plot_scatter_length_vs_aspects(df)

    print_summary_stats(df)


if __name__ == "__main__":
    main()
