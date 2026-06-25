"""
EDA Visualizations for ABSA Telecom Dataset
Generates 4 charts and saves them to outputs/eda/
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

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150


def load_data(path: str) -> pd.DataFrame:
    """Load JSON dataset and parse columns."""
    with open(path, "r") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # Ensure aspects and aspect_sentiments are proper Python objects
    # (they already are from JSON load, but handle string case too)
    if isinstance(df["aspects"].iloc[0], str):
        df["aspects"] = df["aspects"].apply(json.loads)
    if isinstance(df["aspect_sentiments"].iloc[0], str):
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)

    # Add feedback_length column
    df["feedback_length"] = df["feedback"].apply(lambda x: len(x.split()))
    return df


def plot_aspect_frequency(df: pd.DataFrame):
    """1. Horizontal bar chart of aspect frequency (sorted descending)."""
    # Explode aspects list into individual rows
    exploded = df.explode("aspects")
    aspect_counts = exploded["aspects"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(aspect_counts.index, aspect_counts.values, color=sns.color_palette("viridis", len(aspect_counts)))
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Aspect", fontsize=12)
    ax.set_title("Aspect Frequency Distribution", fontsize=14, fontweight="bold")

    # Add count labels on bars
    for bar, val in zip(bars, aspect_counts.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "01_aspect_frequency_bar.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_sentiment_pie(df: pd.DataFrame):
    """2. Overall sentiment distribution pie chart."""
    sentiment_counts = Counter()
    for sent_dict in df["aspect_sentiments"]:
        for sentiment in sent_dict.values():
            sentiment_counts[sentiment] += 1

    labels = ["positive", "negative", "neutral"]
    values = [sentiment_counts[l] for l in labels]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140, textprops={"fontsize": 12}
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title("Overall Sentiment Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "02_sentiment_distribution_pie.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_sentiment_heatmap(df: pd.DataFrame):
    """3. Category-wise sentiment heatmap (aspects x sentiments)."""
    # Build a matrix: rows=aspects, cols=sentiments
    sentiment_labels = ["positive", "negative", "neutral"]
    aspect_sentiment_data = {}

    for _, row in df.iterrows():
        for aspect, sentiment in row["aspect_sentiments"].items():
            if aspect not in aspect_sentiment_data:
                aspect_sentiment_data[aspect] = Counter()
            aspect_sentiment_data[aspect][sentiment] += 1

    # Create DataFrame for heatmap
    aspects_sorted = sorted(aspect_sentiment_data.keys(),
                            key=lambda x: sum(aspect_sentiment_data[x].values()),
                            reverse=True)
    heatmap_data = []
    for aspect in aspects_sorted:
        row = [aspect_sentiment_data[aspect].get(s, 0) for s in sentiment_labels]
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=aspects_sorted, columns=sentiment_labels)

    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Aspect", fontsize=12)
    ax.set_title("Aspect-Sentiment Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "03_aspect_sentiment_heatmap.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_grouped_bar(df: pd.DataFrame):
    """4. Grouped bar chart — positive/negative/neutral per aspect."""
    sentiment_labels = ["positive", "negative", "neutral"]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    aspect_sentiment_data = {}
    for _, row in df.iterrows():
        for aspect, sentiment in row["aspect_sentiments"].items():
            if aspect not in aspect_sentiment_data:
                aspect_sentiment_data[aspect] = Counter()
            aspect_sentiment_data[aspect][sentiment] += 1

    # Sort aspects by total frequency descending
    aspects_sorted = sorted(aspect_sentiment_data.keys(),
                            key=lambda x: sum(aspect_sentiment_data[x].values()),
                            reverse=True)

    # Prepare data for grouped bar
    x = np.arange(len(aspects_sorted))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (sent, color) in enumerate(zip(sentiment_labels, colors)):
        values = [aspect_sentiment_data[a].get(sent, 0) for a in aspects_sorted]
        bars = ax.bar(x + i * width, values, width, label=sent, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(aspects_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Aspect", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sentiment Breakdown by Aspect", fontsize=14, fontweight="bold")
    ax.legend(title="Sentiment", fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "04_grouped_sentiment_bar.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} entries.\n")

    print("Generating visualizations:")
    plot_aspect_frequency(df)
    plot_sentiment_pie(df)
    plot_sentiment_heatmap(df)
    plot_grouped_bar(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
