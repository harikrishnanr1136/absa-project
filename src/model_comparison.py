"""
Model Comparison Script — Full metrics comparison across both models.

Generates formatted tables (tabulate "grid" format) comparing:
  Table 1: Aspect Detection Performance
  Table 2: Sentiment Classification Performance
  Table 3: Hardware Comparison

Saves to outputs/models/metrics_comparison_tables.txt and metrics_comparison.json
"""

import json
import logging
import os

from tabulate import tabulate

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    """Load JSON with error handling."""
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


def format_delta(m1_val, m2_val, higher_is_better=True):
    """
    Compute delta. Positive means Model 2 is better.
    Flag with * if Model 1 outperforms Model 2.
    """
    delta = m2_val - m1_val
    if higher_is_better:
        marker = " *" if delta < 0 else ""
    else:
        # For metrics like hamming loss, lower is better
        marker = " *" if delta > 0 else ""
    return f"{delta:+.4f}{marker}"


def build_aspect_detection_table(m1: dict, m2: dict) -> list:
    """Build Table 1 — Aspect Detection Performance."""
    rows = []
    for split in ["train", "val", "test"]:
        m1_micro = safe_get(m1, "aspect_detection", split, "micro", "f1")
        m2_micro = safe_get(m2, "aspect_detection", split, "micro", "f1")
        rows.append([split.capitalize(), "Micro-F1", f"{m1_micro:.4f}", f"{m2_micro:.4f}",
                     format_delta(m1_micro, m2_micro)])

        m1_macro = safe_get(m1, "aspect_detection", split, "macro", "f1")
        m2_macro = safe_get(m2, "aspect_detection", split, "macro", "f1")
        rows.append([split.capitalize(), "Macro-F1", f"{m1_macro:.4f}", f"{m2_macro:.4f}",
                     format_delta(m1_macro, m2_macro)])

        m1_hl = safe_get(m1, "aspect_detection", split, "hamming_loss")
        m2_hl = safe_get(m2, "aspect_detection", split, "hamming_loss")
        rows.append([split.capitalize(), "Hamming Loss", f"{m1_hl:.4f}", f"{m2_hl:.4f}",
                     format_delta(m1_hl, m2_hl, higher_is_better=False)])

    return rows


def build_sentiment_table(m1: dict, m2: dict) -> list:
    """Build Table 2 — Sentiment Classification Performance (overall across aspects)."""
    rows = []
    for split in ["train", "val", "test"]:
        m1_acc = safe_get(m1, "sentiment_classification", split, "_overall", "accuracy")
        m2_acc = safe_get(m2, "sentiment_classification", split, "_overall", "accuracy")
        rows.append([split.capitalize(), "Accuracy", f"{m1_acc:.4f}", f"{m2_acc:.4f}",
                     format_delta(m1_acc, m2_acc)])

        m1_macro = safe_get(m1, "sentiment_classification", split, "_overall", "macro_f1")
        m2_macro = safe_get(m2, "sentiment_classification", split, "_overall", "macro_f1")
        rows.append([split.capitalize(), "F1 (macro)", f"{m1_macro:.4f}", f"{m2_macro:.4f}",
                     format_delta(m1_macro, m2_macro)])

        m1_weighted = safe_get(m1, "sentiment_classification", split, "_overall", "weighted_f1")
        m2_weighted = safe_get(m2, "sentiment_classification", split, "_overall", "weighted_f1")
        rows.append([split.capitalize(), "F1 (weighted)", f"{m1_weighted:.4f}", f"{m2_weighted:.4f}",
                     format_delta(m1_weighted, m2_weighted)])

    return rows


def build_hardware_table(m1_hw: dict, m2_hw: dict) -> list:
    """Build Table 3 — Hardware Comparison."""
    # Extract values flexibly from nested structures
    m1_train_time = safe_get(m1_hw, "training", "time_seconds")
    m2_train_time = safe_get(m2_hw, "training", "time_seconds",
                             default=safe_get(m2_hw, "training_time_seconds"))

    m1_inference = safe_get(m1_hw, "inference", "per_sample_avg_ms")
    m2_inference = safe_get(m2_hw, "inference", "per_sample_avg_ms",
                            default=safe_get(m2_hw, "inference_time_ms_per_sample"))

    m1_memory = safe_get(m1_hw, "training", "peak_ram_mb")
    m2_memory = safe_get(m2_hw, "peak_inference_memory_mb",
                         default=safe_get(m2_hw, "training", "peak_ram_mb"))

    m1_size = safe_get(m1_hw, "model_size", "total_mb")
    m2_size = 253.0  # DistilBERT ~253 MB

    rows = [
        ["Training time (seconds)", f"{m1_train_time:.3f}", f"{m2_train_time:.3f}",
         format_delta(m1_train_time, m2_train_time, higher_is_better=False)],
        ["Inference time (ms/sample)", f"{m1_inference:.4f}", f"{m2_inference:.4f}",
         format_delta(m1_inference, m2_inference, higher_is_better=False)],
        ["Peak memory (MB)", f"{m1_memory:.3f}", f"{m2_memory:.3f}",
         format_delta(m1_memory, m2_memory, higher_is_better=False)],
        ["Model size on disk (MB)", f"{m1_size:.4f}", f"{m2_size:.1f}",
         format_delta(m1_size, m2_size, higher_is_better=False)],
    ]

    return rows


def main():
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON — Full Metrics Tables")
    logger.info("=" * 70)

    # Load config and paths
    config = load_config()
    output_dir = resolve_path(config["outputs"]["models"])

    # Load metric files
    m1_path = os.path.join(output_dir, "model1_metrics.json")
    m2_path = os.path.join(output_dir, "model2_metrics.json")
    m1_hw_path = os.path.join(output_dir, "model1_hardware_report.json")
    m2_hw_path = os.path.join(output_dir, "model2_hardware_report.json")

    m1 = load_json(m1_path)
    m2 = load_json(m2_path)
    m1_hw = load_json(m1_hw_path)
    m2_hw = load_json(m2_hw_path)

    logger.info(f"Loaded: model1_metrics={'OK' if m1 else 'MISSING'}")
    logger.info(f"Loaded: model2_metrics={'OK' if m2 else 'MISSING'}")
    logger.info(f"Loaded: model1_hardware={'OK' if m1_hw else 'MISSING'}")
    logger.info(f"Loaded: model2_hardware={'OK' if m2_hw else 'MISSING'}")

    # ─── Table 1: Aspect Detection ───────────────────────────────────────
    aspect_rows = build_aspect_detection_table(m1, m2)
    aspect_headers = ["Split", "Metric", "Model 1 (LR+TF-IDF)", "Model 2 (DistilBERT)", "Delta"]
    table1 = tabulate(aspect_rows, headers=aspect_headers, tablefmt="grid")

    # ─── Table 2: Sentiment Classification ────────────────────────────────
    sentiment_rows = build_sentiment_table(m1, m2)
    sentiment_headers = ["Split", "Metric", "Model 1", "Model 2", "Delta"]
    table2 = tabulate(sentiment_rows, headers=sentiment_headers, tablefmt="grid")

    # ─── Table 3: Hardware ────────────────────────────────────────────────
    hw_rows = build_hardware_table(m1_hw, m2_hw)
    hw_headers = ["Property", "Model 1", "Model 2", "Delta"]
    table3 = tabulate(hw_rows, headers=hw_headers, tablefmt="grid")

    # ─── Print ────────────────────────────────────────────────────────────
    full_output = []

    full_output.append("=" * 70)
    full_output.append("TABLE 1 — Aspect Detection Performance")
    full_output.append("(* = Model 1 outperforms Model 2)")
    full_output.append("=" * 70)
    full_output.append(table1)

    full_output.append("")
    full_output.append("=" * 70)
    full_output.append("TABLE 2 — Sentiment Classification Performance")
    full_output.append("(averaged across all aspects, * = Model 1 better)")
    full_output.append("=" * 70)
    full_output.append(table2)

    full_output.append("")
    full_output.append("=" * 70)
    full_output.append("TABLE 3 — Hardware Comparison")
    full_output.append("(lower is better for all rows, * = Model 1 better)")
    full_output.append("=" * 70)
    full_output.append(table3)

    full_text = "\n".join(full_output)
    print(full_text)

    # ─── Save as text ─────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "metrics_comparison_tables.txt")
    with open(txt_path, "w") as f:
        f.write(full_text)
    logger.info(f"\nTables saved: {txt_path}")

    # ─── Save as JSON ─────────────────────────────────────────────────────
    json_data = {
        "aspect_detection": [
            dict(zip(aspect_headers, row)) for row in aspect_rows
        ],
        "sentiment_classification": [
            dict(zip(sentiment_headers, row)) for row in sentiment_rows
        ],
        "hardware": [
            dict(zip(hw_headers, row)) for row in hw_rows
        ],
    }
    json_path = os.path.join(output_dir, "metrics_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"JSON saved: {json_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
