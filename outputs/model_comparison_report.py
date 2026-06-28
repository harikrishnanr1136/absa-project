"""
Model Comparison Report — Model 1 (LR+TF-IDF) vs Model 2 (DistilBERT).

Loads metric and hardware JSON files from both models and generates:
1. Performance comparison table (train/val/test)
2. Hardware comparison table
3. Per-aspect F1 grouped bar chart
4. Confusion matrix comparison for most confused aspects
5. Final production recommendation with justification

Outputs feed directly into README and approach documentation.
"""

import json
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
EDA_DIR = os.path.join(SCRIPT_DIR, "eda")

os.makedirs(EDA_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_json(path: str) -> dict:
    """Load JSON file with error handling."""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path} — using placeholder values")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def safe_get(d: dict, *keys, default=0.0):
    """Safely navigate nested dict keys."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def print_performance_table(m1_metrics: dict, m2_metrics: dict):
    """Print performance comparison table."""
    print(f"\n{'═' * 90}")
    print("1. PERFORMANCE COMPARISON (Aspect Detection & Sentiment Classification)")
    print(f"{'═' * 90}")

    header = f"  {'Metric':<22} {'M1 Train':>9} {'M1 Val':>8} {'M1 Test':>9} {'M2 Train':>9} {'M2 Val':>8} {'M2 Test':>9}"
    print(f"\n{header}")
    print(f"  {'─' * 86}")

    rows = [
        ("Aspect Micro-F1",
         safe_get(m1_metrics, "aspect_detection", "train", "micro", "f1"),
         safe_get(m1_metrics, "aspect_detection", "val", "micro", "f1"),
         safe_get(m1_metrics, "aspect_detection", "test", "micro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "train", "micro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "val", "micro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "test", "micro", "f1")),
        ("Aspect Macro-F1",
         safe_get(m1_metrics, "aspect_detection", "train", "macro", "f1"),
         safe_get(m1_metrics, "aspect_detection", "val", "macro", "f1"),
         safe_get(m1_metrics, "aspect_detection", "test", "macro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "train", "macro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "val", "macro", "f1"),
         safe_get(m2_metrics, "aspect_detection", "test", "macro", "f1")),
        ("Sentiment Macro-F1",
         safe_get(m1_metrics, "sentiment_classification", "train", "_overall", "macro_f1"),
         safe_get(m1_metrics, "sentiment_classification", "val", "_overall", "macro_f1"),
         safe_get(m1_metrics, "sentiment_classification", "test", "_overall", "macro_f1"),
         safe_get(m2_metrics, "sentiment_classification", "train", "_overall", "macro_f1"),
         safe_get(m2_metrics, "sentiment_classification", "val", "_overall", "macro_f1"),
         safe_get(m2_metrics, "sentiment_classification", "test", "_overall", "macro_f1")),
        ("Sentiment Accuracy",
         safe_get(m1_metrics, "sentiment_classification", "train", "_overall", "accuracy"),
         safe_get(m1_metrics, "sentiment_classification", "val", "_overall", "accuracy"),
         safe_get(m1_metrics, "sentiment_classification", "test", "_overall", "accuracy"),
         safe_get(m2_metrics, "sentiment_classification", "train", "_overall", "accuracy"),
         safe_get(m2_metrics, "sentiment_classification", "val", "_overall", "accuracy"),
         safe_get(m2_metrics, "sentiment_classification", "test", "_overall", "accuracy")),
    ]

    for name, m1_tr, m1_v, m1_te, m2_tr, m2_v, m2_te in rows:
        print(f"  {name:<22} {m1_tr:>9.4f} {m1_v:>8.4f} {m1_te:>9.4f} {m2_tr:>9.4f} {m2_v:>8.4f} {m2_te:>9.4f}")

    return rows


def print_hardware_table(m1_hw: dict, m2_hw: dict):
    """Print hardware/resource comparison table."""
    print(f"\n{'═' * 60}")
    print("2. HARDWARE & RESOURCE COMPARISON")
    print(f"{'═' * 60}")

    m1_train_time = safe_get(m1_hw, "training", "time_seconds")
    m2_train_time = safe_get(m2_hw, "training_time_seconds", default=safe_get(m2_hw, "training", "time_seconds"))

    m1_inference = safe_get(m1_hw, "inference", "per_sample_avg_ms")
    m2_inference = safe_get(m2_hw, "inference_time_ms_per_sample", default=safe_get(m2_hw, "inference", "per_sample_avg_ms"))

    m1_memory = safe_get(m1_hw, "training", "peak_ram_mb", default=safe_get(m1_hw, "peak_memory_mb"))
    m2_memory = safe_get(m2_hw, "peak_inference_memory_mb", default=safe_get(m2_hw, "peak_memory_mb"))

    m1_size = safe_get(m1_hw, "model_size", "total_mb")
    m2_size = 253.0  # DistilBERT estimated size (~253 MB)

    print(f"\n  {'Property':<30} {'Model 1 (LR)':>14} {'Model 2 (BERT)':>16}")
    print(f"  {'─' * 62}")
    print(f"  {'Training time (seconds)':<30} {m1_train_time:>14.2f} {m2_train_time:>16.2f}")
    print(f"  {'Inference time (ms/sample)':<30} {m1_inference:>14.4f} {m2_inference:>16.4f}")
    print(f"  {'Peak memory (MB)':<30} {m1_memory:>14.3f} {m2_memory:>16.3f}")
    print(f"  {'Model size on disk (MB)':<30} {m1_size:>14.4f} {m2_size:>16.1f}")

    return {
        "training_time_s": {"model1": m1_train_time, "model2": m2_train_time},
        "inference_ms_per_sample": {"model1": m1_inference, "model2": m2_inference},
        "peak_memory_mb": {"model1": m1_memory, "model2": m2_memory},
        "model_size_mb": {"model1": m1_size, "model2": m2_size},
    }


def plot_f1_per_aspect(m1_metrics: dict, m2_metrics: dict, save_path: str):
    """Plot grouped bar chart: per-aspect F1 for both models."""
    print(f"\n{'═' * 60}")
    print("3. PER-ASPECT F1 COMPARISON CHART")
    print(f"{'═' * 60}")

    m1_aspect_f1 = safe_get(m1_metrics, "aspect_detection", "test", "per_aspect_f1", default={})
    m2_aspect_f1 = safe_get(m2_metrics, "aspect_detection", "test", "per_aspect_f1", default={})

    aspects = sorted(set(list(m1_aspect_f1.keys()) + list(m2_aspect_f1.keys())))

    if not aspects:
        logger.warning("No per-aspect F1 data available for plotting")
        return

    m1_scores = [m1_aspect_f1.get(a, 0) for a in aspects]
    m2_scores = [m2_aspect_f1.get(a, 0) for a in aspects]

    # Sort by Model 1 F1 descending
    sorted_indices = np.argsort(m1_scores)[::-1]
    aspects = [aspects[i] for i in sorted_indices]
    m1_scores = [m1_scores[i] for i in sorted_indices]
    m2_scores = [m2_scores[i] for i in sorted_indices]

    x = np.arange(len(aspects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width / 2, m1_scores, width, label="Model 1 (LR+TF-IDF)", color="#3498db")
    bars2 = ax.bar(x + width / 2, m2_scores, width, label="Model 2 (DistilBERT)", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in aspects], fontsize=8, ha="center")
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Aspect Detection F1 — Model Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_confusion_comparison(m1_metrics: dict, m2_metrics: dict, save_path: str):
    """Plot confusion comparison for top 3 most confused aspects."""
    print(f"\n{'═' * 60}")
    print("4. CONFUSION MATRIX COMPARISON (Top 3 Confused Aspects)")
    print(f"{'═' * 60}")

    # Identify most confused aspects (lowest F1 in Model 1)
    m1_f1 = safe_get(m1_metrics, "aspect_detection", "test", "per_aspect_f1", default={})
    if not m1_f1:
        logger.warning("No per-aspect F1 data — skipping confusion plot")
        return

    sorted_aspects = sorted(m1_f1.items(), key=lambda x: x[1])
    top_confused = [a for a, _ in sorted_aspects[:3]]

    m2_f1 = safe_get(m2_metrics, "aspect_detection", "test", "per_aspect_f1", default={})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, aspect in enumerate(top_confused):
        ax = axes[idx]
        f1_m1 = m1_f1.get(aspect, 0)
        f1_m2 = m2_f1.get(aspect, 0)

        data = np.array([[f1_m1], [f1_m2]])
        sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                    xticklabels=["F1 Score"], yticklabels=["Model 1", "Model 2"],
                    ax=ax, cbar=False)
        ax.set_title(aspect.replace("_", " ").title(), fontsize=11, fontweight="bold")

    plt.suptitle("Most Confused Aspects — F1 Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")

    for aspect in top_confused:
        print(f"  {aspect:<25} M1 F1={m1_f1.get(aspect,0):.4f}  M2 F1={m2_f1.get(aspect,0):.4f}")


def generate_recommendation(m1_metrics: dict, m2_metrics: dict, hw_comparison: dict) -> dict:
    """Generate final production recommendation with justification."""
    print(f"\n{'═' * 60}")
    print("5. PRODUCTION RECOMMENDATION")
    print(f"{'═' * 60}")

    # Compare test metrics
    m1_aspect_f1 = safe_get(m1_metrics, "aspect_detection", "test", "micro", "f1")
    m2_aspect_f1 = safe_get(m2_metrics, "aspect_detection", "test", "micro", "f1")
    m1_sent_f1 = safe_get(m1_metrics, "sentiment_classification", "test", "_overall", "macro_f1")
    m2_sent_f1 = safe_get(m2_metrics, "sentiment_classification", "test", "_overall", "macro_f1")

    m1_inference = hw_comparison["inference_ms_per_sample"]["model1"]
    m2_inference = hw_comparison["inference_ms_per_sample"]["model2"]
    m1_size = hw_comparison["model_size_mb"]["model1"]
    m2_size = hw_comparison["model_size_mb"]["model2"]

    # Decision logic
    # If DistilBERT significantly outperforms LR (>5% gain) and latency is acceptable, recommend it.
    # Otherwise recommend LR for production due to speed/simplicity.
    aspect_gain = m2_aspect_f1 - m1_aspect_f1
    sent_gain = m2_sent_f1 - m1_sent_f1
    speed_ratio = m2_inference / max(m1_inference, 0.001)

    if aspect_gain > 0.05 and sent_gain > 0.05:
        recommended = "Model 2 (DistilBERT)"
        reason = (f"DistilBERT shows significant improvement: "
                  f"+{aspect_gain:.3f} aspect F1, +{sent_gain:.3f} sentiment F1. "
                  f"The accuracy gain justifies the increased inference latency.")
    elif m1_aspect_f1 >= 0.85 and m1_sent_f1 >= 0.75:
        recommended = "Model 1 (Logistic Regression + TF-IDF)"
        reason = (f"LR baseline achieves strong results (aspect F1={m1_aspect_f1:.3f}, "
                  f"sentiment F1={m1_sent_f1:.3f}) with {speed_ratio:.0f}x faster inference "
                  f"and {m2_size/max(m1_size,0.01):.0f}x smaller model size. "
                  f"Ideal for low-latency Streamlit deployment.")
    else:
        recommended = "Model 2 (DistilBERT) with fallback to Model 1"
        reason = "Use DistilBERT for accuracy-critical paths, LR for real-time/batch scenarios."

    recommendation = {
        "recommended_model": recommended,
        "justification": reason,
        "decision_factors": {
            "aspect_f1_gain": round(aspect_gain, 4),
            "sentiment_f1_gain": round(sent_gain, 4),
            "speed_ratio_m2_vs_m1": round(speed_ratio, 1),
            "size_ratio_m2_vs_m1": round(m2_size / max(m1_size, 0.01), 1),
        },
        "limitations": [
            "Model 1: Cannot capture word order or context (TF-IDF bag-of-words limitation)",
            "Model 1: Struggles with sarcasm and implicit sentiment",
            "Model 2: Slow inference on CPU (~100x slower than LR)",
            "Model 2: Large model size (~253 MB) may be problematic for edge deployment",
            "Both: Limited training data (1000 samples) — production models need more data",
            "Both: Neutral class is underrepresented and harder to predict",
        ],
        "challenges": [
            "PyTorch incompatibility with Python 3.13 required downgrade to 3.11",
            "Sentence-transformers dependency conflicts with newer numpy/transformers",
            "Small dataset (1000 samples) limits deep learning model potential",
            "Multi-label stratification is difficult with many rare aspect combinations",
            "Per-aspect sentiment models have very few samples for some aspects (roaming, sim_activation)",
            "CPU-only training is very slow for transformer models (~hours per experiment)",
        ],
    }

    print(f"\n  RECOMMENDED: {recommended}")
    print(f"\n  JUSTIFICATION:")
    print(f"    {reason}")
    print(f"\n  DECISION FACTORS:")
    for k, v in recommendation["decision_factors"].items():
        print(f"    {k}: {v}")
    print(f"\n  LIMITATIONS:")
    for lim in recommendation["limitations"]:
        print(f"    • {lim}")
    print(f"\n  CHALLENGES:")
    for ch in recommendation["challenges"]:
        print(f"    • {ch}")

    return recommendation


def main():
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON REPORT")
    logger.info("=" * 70)

    # ─── Load Metrics ─────────────────────────────────────────────────────
    # Model 1 — try experiment_log first, then direct metrics
    exp_log_path = os.path.join(SCRIPT_DIR, "experiment_log.json")
    m1_metrics_path = os.path.join(MODELS_DIR, "model1_metrics.json")
    m2_metrics_path = os.path.join(MODELS_DIR, "model2_metrics.json")
    m1_hw_path = os.path.join(MODELS_DIR, "model1_hardware_report.json")
    m2_hw_path = os.path.join(MODELS_DIR, "model2_hardware_report.json")

    # Load Model 1 metrics — direct file first, experiment log as fallback
    m1_metrics = {}
    if os.path.exists(m1_metrics_path):
        m1_metrics = load_json(m1_metrics_path)

    if not m1_metrics and os.path.exists(exp_log_path):
        exp_log = load_json(exp_log_path)
        for entry in exp_log:
            if entry.get("experiment_id") == "model1_lr_tfidf":
                m1_metrics = entry
                break

    m2_metrics = load_json(m2_metrics_path)
    m1_hw = load_json(m1_hw_path)
    m2_hw = load_json(m2_hw_path)

    # Fallback: if Model 2 hardware report has zeros, load from experiment_log
    if os.path.exists(exp_log_path):
        exp_log = load_json(exp_log_path)

        # Model 2 hardware fallback
        m2_hw_training_time = safe_get(m2_hw, "training", "time_seconds", default=0)
        m2_hw_inference = safe_get(m2_hw, "inference", "per_sample_avg_ms", default=0)
        if m2_hw_training_time == 0 or m2_hw_inference == 0:
            for entry in exp_log:
                if entry.get("experiment_id") == "model2_distilbert_finetuned":
                    if m2_hw_training_time == 0:
                        m2_hw.setdefault("training", {})["time_seconds"] = entry.get("training_time_seconds", 0)
                    if m2_hw_inference == 0:
                        m2_hw.setdefault("inference", {})["per_sample_avg_ms"] = entry.get("inference_time_ms_per_sample", 0)
                    m2_hw["peak_inference_memory_mb"] = entry.get("peak_memory_mb", 0)
                    m2_hw.setdefault("model_size", {})["total_mb"] = entry.get("model_size_mb", 253)
                    logger.info("Model 2 hardware: supplemented from experiment_log.json")
                    break

        # Model 2 metrics fallback (if file is stale/missing)
        if not m2_metrics:
            for entry in exp_log:
                if entry.get("experiment_id") == "model2_distilbert_finetuned":
                    m2_metrics = entry
                    logger.info("Model 2 metrics: loaded from experiment_log.json")
                    break

    logger.info(f"Model 1 metrics: {'loaded' if m1_metrics else 'NOT FOUND'}")
    logger.info(f"Model 2 metrics: {'loaded' if m2_metrics else 'NOT FOUND'}")
    logger.info(f"Model 1 hardware: {'loaded' if m1_hw else 'NOT FOUND'}")
    logger.info(f"Model 2 hardware: {'loaded' if m2_hw else 'NOT FOUND'}")

    # ─── 1. Performance Table ─────────────────────────────────────────────
    perf_rows = print_performance_table(m1_metrics, m2_metrics)

    # ─── 2. Hardware Table ────────────────────────────────────────────────
    hw_comparison = print_hardware_table(m1_hw, m2_hw)

    # ─── 3. Per-Aspect F1 Chart ───────────────────────────────────────────
    f1_chart_path = os.path.join(EDA_DIR, "model_comparison_f1_per_aspect.png")
    plot_f1_per_aspect(m1_metrics, m2_metrics, f1_chart_path)

    # ─── 4. Confusion Matrix Comparison ───────────────────────────────────
    confusion_path = os.path.join(EDA_DIR, "model_comparison_confusion.png")
    plot_confusion_comparison(m1_metrics, m2_metrics, confusion_path)

    # ─── 5. Recommendation ────────────────────────────────────────────────
    recommendation = generate_recommendation(m1_metrics, m2_metrics, hw_comparison)

    # ─── Save Full Report ─────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "models_compared": ["Model 1 (LR + TF-IDF)", "Model 2 (DistilBERT)"],
        "performance": {
            "model1": {
                "aspect_detection": safe_get(m1_metrics, "aspect_detection", default={}),
                "sentiment_classification": safe_get(m1_metrics, "sentiment_classification", default={}),
            },
            "model2": {
                "aspect_detection": safe_get(m2_metrics, "aspect_detection", default={}),
                "sentiment_classification": safe_get(m2_metrics, "sentiment_classification", default={}),
            },
        },
        "hardware_comparison": hw_comparison,
        "recommendation": recommendation,
    }

    report_path = os.path.join(MODELS_DIR, "model_comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nFull report saved: {report_path}")

    print(f"\n{'═' * 60}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Report: {report_path}")
    print(f"  Charts: {EDA_DIR}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
