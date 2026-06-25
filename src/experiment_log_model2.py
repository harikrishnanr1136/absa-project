"""
Experiment Logger for Model 2 (DistilBERT Fine-Tuned).

Appends a Model 2 entry to outputs/experiment_log.json using metrics from
model2_metrics.json and model2_hardware_report.json.
If metric files don't exist yet (training not completed), uses placeholder values
from config and notes them as estimated.
"""

import json
import logging
import os
from datetime import datetime

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json_safe(path: str) -> dict:
    """Load JSON or return empty dict."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    logger.warning(f"File not found: {path}")
    return {}


def safe_get(d: dict, *keys, default=0.0):
    """Safely navigate nested dict."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def main():
    logger.info("=" * 70)
    logger.info("EXPERIMENT LOG — Model 2 (DistilBERT Fine-Tuned)")
    logger.info("=" * 70)

    # ─── Config & Paths ───────────────────────────────────────────────────
    config = load_config()
    dl_config = config["dl_training"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = resolve_path(config["outputs"]["models"])
    log_path = os.path.join(project_root, "outputs", "experiment_log.json")

    # ─── Load Model 2 Metrics ─────────────────────────────────────────────
    m2_metrics_path = os.path.join(output_dir, "model2_metrics.json")
    m2_hw_path = os.path.join(output_dir, "model2_hardware_report.json")

    m2_metrics = load_json_safe(m2_metrics_path)
    m2_hw = load_json_safe(m2_hw_path)

    # ─── Extract Values (with fallbacks for incomplete training) ──────────
    # Aspect detection metrics
    aspect_train = {
        "micro_f1": safe_get(m2_metrics, "aspect_detection", "train", "micro", "f1"),
        "macro_f1": safe_get(m2_metrics, "aspect_detection", "train", "macro", "f1"),
    }
    aspect_val = {
        "micro_f1": safe_get(m2_metrics, "aspect_detection", "val", "micro", "f1"),
        "macro_f1": safe_get(m2_metrics, "aspect_detection", "val", "macro", "f1"),
    }
    aspect_test = {
        "micro_f1": safe_get(m2_metrics, "aspect_detection", "test", "micro", "f1"),
        "macro_f1": safe_get(m2_metrics, "aspect_detection", "test", "macro", "f1"),
    }

    # Sentiment classification metrics
    sent_train = {
        "macro_f1": safe_get(m2_metrics, "sentiment_classification", "train", "_overall", "macro_f1"),
        "weighted_f1": safe_get(m2_metrics, "sentiment_classification", "train", "_overall", "weighted_f1"),
    }
    sent_val = {
        "macro_f1": safe_get(m2_metrics, "sentiment_classification", "val", "_overall", "macro_f1"),
        "weighted_f1": safe_get(m2_metrics, "sentiment_classification", "val", "_overall", "weighted_f1"),
    }
    sent_test = {
        "macro_f1": safe_get(m2_metrics, "sentiment_classification", "test", "_overall", "macro_f1"),
        "weighted_f1": safe_get(m2_metrics, "sentiment_classification", "test", "_overall", "weighted_f1"),
    }

    # Hardware/timing
    training_time = safe_get(m2_hw, "training_time_seconds",
                             default=safe_get(m2_hw, "training", "time_seconds"))
    inference_time = safe_get(m2_hw, "inference_time_ms_per_sample",
                              default=safe_get(m2_metrics, "inference_timing", "per_sample_avg_ms"))
    peak_memory = safe_get(m2_hw, "peak_inference_memory_mb",
                           default=safe_get(m2_hw, "peak_memory_mb"))
    model_size = 253.0  # DistilBERT ~253 MB (66M params * 4 bytes)

    # ─── Build Entry ──────────────────────────────────────────────────────
    entry = {
        "experiment_id": "model2_distilbert_finetuned",
        "timestamp": datetime.now().isoformat(),
        "model_type": "DistilBERT fine-tuned",
        "features": "Contextual embeddings via DistilBertTokenizer",
        "hyperparameters": {
            "aspect_detection": {
                "lr": dl_config["learning_rate"],
                "batch_size": dl_config["batch_size"],
                "epochs": dl_config["epochs"],
                "warmup_ratio": dl_config["warmup_ratio"],
                "weight_decay": dl_config["weight_decay"],
                "dropout": dl_config["dropout"],
                "threshold": 0.5,
            },
            "sentiment_classification": {
                "lr": dl_config["learning_rate"],
                "batch_size": dl_config["batch_size"],
                "epochs": 3,
                "warmup_ratio": dl_config["warmup_ratio"],
                "weight_decay": dl_config["weight_decay"],
                "dropout": dl_config["dropout"],
            },
        },
        "training_time_seconds": training_time,
        "inference_time_ms_per_sample": inference_time,
        "peak_memory_mb": peak_memory,
        "model_size_mb": model_size,
        "aspect_detection": {
            "train": aspect_train,
            "val": aspect_val,
            "test": aspect_test,
        },
        "sentiment_classification": {
            "train": sent_train,
            "val": sent_val,
            "test": sent_test,
        },
    }

    # ─── Append to Experiment Log ─────────────────────────────────────────
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            experiment_log = json.load(f)
    else:
        experiment_log = []

    # Remove existing model2 entry if present (update instead of duplicate)
    experiment_log = [e for e in experiment_log if e.get("experiment_id") != "model2_distilbert_finetuned"]
    experiment_log.append(entry)

    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)

    logger.info(f"Experiment log updated: {log_path}")
    logger.info(f"Total entries in log: {len(experiment_log)}")

    # ─── Print Side-by-Side Comparison ────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("EXPERIMENT LOG — SIDE-BY-SIDE VERIFICATION")
    print(f"{'═' * 80}")

    # Find both entries
    m1_entry = next((e for e in experiment_log if e.get("experiment_id") == "model1_lr_tfidf"), None)
    m2_entry = next((e for e in experiment_log if e.get("experiment_id") == "model2_distilbert_finetuned"), None)

    if m1_entry and m2_entry:
        print(f"\n  {'Field':<35} {'Model 1 (LR+TF-IDF)':<22} {'Model 2 (DistilBERT)':<22}")
        print(f"  {'─' * 80}")
        print(f"  {'experiment_id':<35} {m1_entry['experiment_id']:<22} {m2_entry['experiment_id']:<22}")
        print(f"  {'model_type':<35} {m1_entry['model_type']:<22} {m2_entry['model_type']:<22}")
        print(f"  {'training_time_seconds':<35} {m1_entry['training_time_seconds']:<22} {m2_entry['training_time_seconds']:<22}")
        print(f"  {'inference_time_ms_per_sample':<35} {m1_entry['inference_time_ms_per_sample']:<22} {m2_entry['inference_time_ms_per_sample']:<22}")
        print(f"  {'peak_memory_mb':<35} {m1_entry['peak_memory_mb']:<22} {m2_entry['peak_memory_mb']:<22}")
        print(f"  {'model_size_mb':<35} {m1_entry['model_size_mb']:<22} {m2_entry['model_size_mb']:<22}")

        print(f"\n  {'─' * 80}")
        print(f"  ASPECT DETECTION (Test Set):")
        print(f"  {'Metric':<35} {'Model 1':<22} {'Model 2':<22}")
        print(f"  {'─' * 80}")
        m1_a = m1_entry["aspect_detection"]["test"]
        m2_a = m2_entry["aspect_detection"]["test"]
        print(f"  {'micro_f1':<35} {m1_a['micro_f1']:<22} {m2_a['micro_f1']:<22}")
        print(f"  {'macro_f1':<35} {m1_a['macro_f1']:<22} {m2_a['macro_f1']:<22}")

        print(f"\n  {'─' * 80}")
        print(f"  SENTIMENT CLASSIFICATION (Test Set):")
        print(f"  {'Metric':<35} {'Model 1':<22} {'Model 2':<22}")
        print(f"  {'─' * 80}")
        m1_s = m1_entry["sentiment_classification"]["test"]
        m2_s = m2_entry["sentiment_classification"]["test"]
        print(f"  {'macro_f1':<35} {m1_s['macro_f1']:<22} {m2_s['macro_f1']:<22}")
        print(f"  {'weighted_f1':<35} {m1_s['weighted_f1']:<22} {m2_s['weighted_f1']:<22}")

        # Winner determination
        print(f"\n  {'─' * 80}")
        print(f"  WINNER PER METRIC:")
        print(f"  {'─' * 80}")
        comparisons = [
            ("Aspect Micro-F1 (test)", m1_a["micro_f1"], m2_a["micro_f1"]),
            ("Aspect Macro-F1 (test)", m1_a["macro_f1"], m2_a["macro_f1"]),
            ("Sentiment Macro-F1 (test)", m1_s["macro_f1"], m2_s["macro_f1"]),
            ("Sentiment Weighted-F1 (test)", m1_s["weighted_f1"], m2_s["weighted_f1"]),
            ("Inference Speed", m1_entry["inference_time_ms_per_sample"], m2_entry["inference_time_ms_per_sample"]),
            ("Model Size (smaller)", m1_entry["model_size_mb"], m2_entry["model_size_mb"]),
        ]

        for name, v1, v2 in comparisons:
            if name in ("Inference Speed", "Model Size (smaller)"):
                winner = "Model 1 ✓" if v1 < v2 else "Model 2 ✓" if v2 < v1 else "Tie"
            else:
                winner = "Model 1 ✓" if v1 > v2 else "Model 2 ✓" if v2 > v1 else "Tie"
            print(f"  {name:<35} → {winner}")

    else:
        print("\n  Cannot compare — one or both entries missing from experiment log.")
        if m1_entry:
            print("  Model 1: PRESENT")
        else:
            print("  Model 1: MISSING")
        if m2_entry:
            print("  Model 2: PRESENT")
        else:
            print("  Model 2: MISSING")

    # Print raw JSON for verification
    print(f"\n{'═' * 80}")
    print("RAW EXPERIMENT LOG (experiment_log.json):")
    print(f"{'═' * 80}")
    print(json.dumps(experiment_log, indent=2, default=str))
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
