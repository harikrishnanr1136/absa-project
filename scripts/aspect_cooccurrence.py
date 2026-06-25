"""
Aspect Co-occurrence Analysis for ABSA Telecom Dataset
Builds a co-occurrence matrix and visualizes as a heatmap.
"""

import json
import os
from itertools import combinations

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

sns.set_theme(style="white")
plt.rcParams["figure.dpi"] = 150

ASPECTS = [
    "network_coverage", "internet_speed", "call_quality", "customer_support",
    "billing", "recharge_plans", "data_balance", "roaming", "sim_activation",
    "mobile_app_experience", "ott_bundle_services", "pricing",
    "value_for_money", "data_validity", "5g_experience",
]


def load_data(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_cooccurrence_matrix(data: list) -> pd.DataFrame:
    """Build a 15x15 symmetric co-occurrence matrix."""
    n = len(ASPECTS)
    matrix = np.zeros((n, n), dtype=int)
    aspect_to_idx = {a: i for i, a in enumerate(ASPECTS)}

    for entry in data:
        aspects_in_entry = [a for a in entry["aspects"] if a in aspect_to_idx]
        # Count all pairs
        for a1, a2 in combinations(aspects_in_entry, 2):
            i, j = aspect_to_idx[a1], aspect_to_idx[a2]
            matrix[i][j] += 1
            matrix[j][i] += 1

    # Diagonal = self-occurrence (total count of each aspect)
    for entry in data:
        for a in entry["aspects"]:
            if a in aspect_to_idx:
                matrix[aspect_to_idx[a]][aspect_to_idx[a]] += 1

    return pd.DataFrame(matrix, index=ASPECTS, columns=ASPECTS)


def plot_heatmap(cooc_df: pd.DataFrame):
    """Visualize co-occurrence matrix as heatmap (diagonal masked)."""
    # Mask diagonal
    mask = np.eye(len(ASPECTS), dtype=bool)

    # Use short labels for readability
    short_labels = [a.replace("_", " ").replace("experience", "exp").replace("services", "svc")
                    for a in ASPECTS]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cooc_df,
        mask=mask,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Co-occurrence Count"},
        square=True,
    )

    ax.set_title("Aspect Co-occurrence Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "cooccurrence_heatmap.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def get_top_pairs(cooc_df: pd.DataFrame, top_n: int = 10) -> list:
    """Extract top N most frequently co-occurring aspect pairs."""
    pairs = []
    n = len(ASPECTS)

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((ASPECTS[i], ASPECTS[j], cooc_df.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def main():
    print("Loading dataset...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} entries.\n")

    # Build matrix
    print("Building co-occurrence matrix...")
    cooc_df = build_cooccurrence_matrix(data)

    # Plot
    print("Generating heatmap:")
    plot_heatmap(cooc_df)

    # Top pairs table
    top_pairs = get_top_pairs(cooc_df, top_n=10)

    print(f"\n{'=' * 65}")
    print("TOP 10 MOST FREQUENTLY CO-OCCURRING ASPECT PAIRS")
    print(f"{'=' * 65}")
    print(f"\n  {'Rank':<6} {'Aspect 1':<25} {'Aspect 2':<25} {'Count':>6}")
    print(f"  {'─' * 63}")

    for rank, (a1, a2, count) in enumerate(top_pairs, 1):
        print(f"  {rank:<6} {a1:<25} {a2:<25} {count:>6}")

    print(f"  {'─' * 63}")

    # Additional stats
    total_multi = sum(1 for entry in data if len(entry["aspects"]) >= 2)
    print(f"\n  Total entries with 2+ aspects: {total_multi}")
    print(f"  Total unique co-occurring pairs with count > 0: "
          f"{sum(1 for a1, a2, c in get_top_pairs(cooc_df, top_n=200) if c > 0)}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
