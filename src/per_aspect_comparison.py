"""
Per-Aspect Performance Comparison — Model 1 vs Model 2.

Generates detailed per-aspect F1 tables, grouped bar charts, and identifies
strengths/weaknesses of each model at the aspect level.
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_json(path: str) -> dict:
    """Load JSON file."""
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def safe_get(d: dict, *keys, default=0.0):
    """Safely navigate nested dict."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def build_per_aspect_table(m1: dict, m2: dict, aspect_labels: list) -> list:
    """
    Build per-aspect F1 comparison table from test set metrics.
    Returns list of row dicts.
    """
    m1_aspect_f1 = safe_get(m1, "aspect_detection", "test", "per_aspect_f1", default={})
    m2_aspect_f1 = safe_get(m2, "aspect_detection", "test", "per_aspect_f1", default={})

    rows = []
    for aspect in aspect_labels:
        f1_m1 = m1_aspect_f1.get(aspect, 0.0)
        f1_m2 = m2_aspect_f1.get(aspect, 0.0)
        delta = f1_m2 - f1_m1
        winner = "Model 2" if delta > 0 else "Model 1" if delta < 0 else "Tie"

        rows.append({
            "aspect": aspect,
            "model1_f1": round(f1_m1, 4),
            "model2_f1": round(f1_m2, 4),
            "winner": winner,
            "delta": round(delta, 4),
        })

    # Average row
    avg_m1 = np.mean([r["model1_f1"] for r in rows])
    avg_m2 = np.mean([r["model2_f1"] for r in rows])
    avg_delta = avg_m2 - avg_m1
    rows.append({
        "aspect": "AVERAGE",
        "model1_f1": round(avg_m1, 4),
        "model2_f1": round(avg_m2, 4),
        "winner": "Model 2" if avg_delta > 0 else "Model 1",
        "delta": round(avg_delta, 4),
    })

    return rows


def print_table(rows: list):
    """Print formatted table using tabulate."""
    table_data = [
        [r["aspect"], f"{r['model1_f1']:.4f}", f"{r['model2_f1']:.4f}",
         r["winner"], f"{r['delta']:+.4f}"]
        for r in rows
    ]
    headers = ["Aspect", "Model 1 F1", "Model 2 F1", "Winner", "Delta"]
    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_grouped_bar(rows: list, save_path: str):
    """
    Grouped horizontal bar chart: Model 1 vs Model 2 F1 per aspect.
    """
    # Exclude AVERAGE row
    data_rows = [r for r in rows if r["aspect"] != "AVERAGE"]

    # Sort by Model 1 F1 ascending (so highest is at top)
    data_rows.sort(key=lambda r: r["model1_f1"])

    aspects = [r["aspect"].replace("_", " ") for r in data_rows]
    m1_scores = [r["model1_f1"] for r in data_rows]
    m2_scores = [r["model2_f1"] for r in data_rows]

    y = np.arange(len(aspects))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 9))

    bars1 = ax.barh(y - height / 2, m1_scores, height, label="Model 1 (LR+TF-IDF)",
                    color="#3498db", edgecolor="white", linewidth=0.5)
    bars2 = ax.barh(y + height / 2, m2_scores, height, label="Model 2 (DistilBERT)",
                    color="#e67e22", edgecolor="white", linewidth=0.5)

    # Threshold line at F1=0.7
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="F1=0.7 threshold")

    ax.set_yticks(y)
    ax.set_yticklabels(aspects, fontsize=9)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Aspect F1 Comparison — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        if width > 0.05:
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va="center", fontsize=7, color="#2c3e50")
    for bar in bars2:
        width = bar.get_width()
        if width > 0.05:
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va="center", fontsize=7, color="#e67e22")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


def analyze_results(rows: list):
    """Identify strengths, weaknesses, and areas needing improvement."""
    data_rows = [r for r in rows if r["aspect"] != "AVERAGE"]

    # Aspects where Model 2 outperforms by > 0.1
    m2_wins_big = [r for r in data_rows if r["delta"] > 0.1]

    # Aspects where Model 1 outperforms (any margin)
    m1_wins = [r for r in data_rows if r["delta"] < 0]

    # Aspects below F1=0.7 for BOTH models
    both_low = [r for r in data_rows if r["model1_f1"] < 0.7 and r["model2_f1"] < 0.7]

    # Best/worst overall (max of both models)
    best = max(data_rows, key=lambda r: max(r["model1_f1"], r["model2_f1"]))
    worst = min(data_rows, key=lambda r: max(r["model1_f1"], r["model2_f1"]))

    logger.info(f"\n{'═' * 70}")
    logger.info("ANALYSIS")
    logger.info(f"{'═' * 70}")

    logger.info(f"\n  Aspects where Model 2 outperforms Model 1 by > 0.1:")
    if m2_wins_big:
        for r in m2_wins_big:
            logger.info(f"    • {r['aspect']:<28} (delta: {r['delta']:+.4f})")
    else:
        logger.info(f"    None")

    logger.info(f"\n  Aspects where Model 1 outperforms Model 2:")
    if m1_wins:
        for r in sorted(m1_wins, key=lambda r: r["delta"]):
            logger.info(f"    • {r['aspect']:<28} (delta: {r['delta']:+.4f})")
    else:
        logger.info(f"    None")

    logger.info(f"\n  Aspects below F1=0.7 for BOTH models (needs improvement):")
    if both_low:
        for r in both_low:
            logger.info(f"    • {r['aspect']:<28} (M1={r['model1_f1']:.4f}, M2={r['model2_f1']:.4f})")
    else:
        logger.info(f"    None — all aspects have F1 >= 0.7 in at least one model")

    logger.info(f"\n  Best performing aspect:  {best['aspect']} "
                f"(max F1={max(best['model1_f1'], best['model2_f1']):.4f})")
    logger.info(f"  Worst performing aspect: {worst['aspect']} "
                f"(max F1={max(worst['model1_f1'], worst['model2_f1']):.4f})")

    return {
        "model2_wins_big": [r["aspect"] for r in m2_wins_big],
        "model1_wins": [r["aspect"] for r in m1_wins],
        "both_below_threshold": [r["aspect"] for r in both_low],
        "best_aspect": best["aspect"],
        "worst_aspect": worst["aspect"],
    }


def main():
    logger.info("=" * 70)
    logger.info("PER-ASPECT PERFORMANCE COMPARISON")
    logger.info("=" * 70)

    # Config and paths
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(eda_dir, exist_ok=True)

    # Load metrics
    m1 = load_json(os.path.join(output_dir, "model1_metrics.json"))
    m2 = load_json(os.path.join(output_dir, "model2_metrics.json"))

    if not m1 or not m2:
        logger.error("Cannot proceed — metric files missing")
        return

    # ─── Table 1: Per-Aspect F1 ───────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("TABLE: Per-Aspect F1 Comparison (Test Set — Aspect Detection)")
    print(f"{'═' * 70}\n")

    rows = build_per_aspect_table(m1, m2, aspect_labels)
    print_table(rows)

    # ─── Chart ────────────────────────────────────────────────────────────
    chart_path = os.path.join(eda_dir, "per_aspect_f1_comparison.png")
    plot_grouped_bar(rows, chart_path)

    # ─── Analysis ─────────────────────────────────────────────────────────
    analysis = analyze_results(rows)

    # ─── Save JSON ────────────────────────────────────────────────────────
    output_data = {
        "per_aspect_f1": rows,
        "analysis": analysis,
    }
    json_path = os.path.join(output_dir, "per_aspect_comparison.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nJSON saved: {json_path}")

    print(f"\n{'═' * 70}")
    print("DONE")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
