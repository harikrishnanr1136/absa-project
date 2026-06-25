"""
Model 1 (Logistic Regression) Performance & Hardware Report.

Measures training time, inference time, RAM usage, model size, and hardware info.
Saves report to outputs/models/model1_hardware_report.json.
"""

import json
import logging
import os
import platform
import time
import tracemalloc
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.preprocessing import PreprocessingPipeline
from src.features import TFIDFFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def measure_inference_time(model, X_single, X_full, n_repeats: int = 100) -> dict:
    """
    Measure inference time for single sample, full test set, and per-sample average.

    Args:
        model: Fitted sklearn model
        X_single: Single sample feature matrix (1, n_features)
        X_full: Full test set feature matrix (n_samples, n_features)
        n_repeats: Number of repetitions for single sample timing

    Returns:
        Dict with timing measurements in milliseconds
    """
    # Single sample inference (averaged over n_repeats)
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(X_single)
    single_total = (time.perf_counter() - start) * 1000  # ms
    single_avg_ms = single_total / n_repeats

    # Full test set inference
    start = time.perf_counter()
    model.predict(X_full)
    full_set_ms = (time.perf_counter() - start) * 1000

    # Per-sample average on full set
    n_samples = X_full.shape[0]
    per_sample_avg_ms = full_set_ms / n_samples

    return {
        "single_sample_ms": round(single_avg_ms, 4),
        "full_test_set_ms": round(full_set_ms, 4),
        "per_sample_avg_ms": round(per_sample_avg_ms, 4),
        "test_set_size": n_samples,
        "single_sample_repeats": n_repeats,
    }


def measure_training_with_memory(X_train, Y_train, seed: int) -> dict:
    """
    Re-run training to measure time and memory precisely.

    Returns:
        Dict with training_time_seconds and memory stats
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    base_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs"
    )
    model = OneVsRestClassifier(base_clf, n_jobs=-1)

    # Measure memory
    tracemalloc.start()
    start = time.time()
    model.fit(X_train, Y_train)
    training_time = time.time() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "training_time_seconds": round(training_time, 3),
        "peak_memory_mb": round(peak_memory / (1024 * 1024), 3),
    }


def measure_inference_memory(model, X_test) -> float:
    """Measure peak memory during inference."""
    tracemalloc.start()
    model.predict(X_test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / (1024 * 1024), 3)


def get_hardware_info() -> dict:
    """Collect hardware and platform information."""
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count(),
        "gpu_available": gpu_available,
    }


def main():
    logger.info("=" * 70)
    logger.info("MODEL 1 — HARDWARE & PERFORMANCE REPORT")
    logger.info("=" * 70)

    # ─── Load Config & Data ───────────────────────────────────────────────
    config = load_config()
    seed = config["seed"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df["aspects"] = test_df["aspects"].apply(json.loads)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df["aspects"] = train_df["aspects"].apply(json.loads)

    # ─── Load saved model & vectorizer ────────────────────────────────────
    model_path = os.path.join(output_dir, "aspect_detector_lr.pkl")
    tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    mlb_path = os.path.join(output_dir, "mlb.pkl")
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.joblib")

    model = joblib.load(model_path)
    tfidf = TFIDFFeatures()
    tfidf.load(tfidf_path)
    mlb = joblib.load(mlb_path)

    logger.info("Model and vectorizer loaded")

    # ─── Preprocess test data ─────────────────────────────────────────────
    pipeline = PreprocessingPipeline()
    pipeline.load(pipeline_path)

    logger.info("Preprocessing test data...")
    test_texts = pipeline.transform(test_df["feedback"].tolist())
    X_test = tfidf.transform(test_texts)

    logger.info("Preprocessing train data for re-training measurement...")
    train_texts = pipeline.transform(train_df["feedback"].tolist())
    X_train = tfidf.transform(train_texts)
    Y_train = mlb.transform(train_df["aspects"])

    X_single = X_test[0:1]

    # ─── Measurements ─────────────────────────────────────────────────────
    logger.info("Measuring training time and memory...")
    train_metrics = measure_training_with_memory(X_train, Y_train, seed)

    logger.info("Measuring inference time...")
    inference_metrics = measure_inference_time(model, X_single, X_test)

    logger.info("Measuring inference memory...")
    inference_ram_mb = measure_inference_memory(model, X_test)

    hardware = get_hardware_info()

    # Model sizes on disk
    model_sizes = {
        "aspect_detector_lr.pkl": get_file_size_mb(model_path),
        "tfidf_vectorizer.joblib": get_file_size_mb(tfidf_path),
        "mlb.pkl": get_file_size_mb(mlb_path),
        "preprocessing_pipeline.joblib": get_file_size_mb(pipeline_path),
    }
    total_model_size_mb = sum(model_sizes.values())

    # ─── Compile Report ───────────────────────────────────────────────────
    report = {
        "model": "Model 1 — Logistic Regression (OneVsRest, Aspect Detection)",
        "training": {
            "time_seconds": train_metrics["training_time_seconds"],
            "peak_ram_mb": train_metrics["peak_memory_mb"],
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1],
            "classes": Y_train.shape[1],
        },
        "inference": {
            "single_sample_ms": inference_metrics["single_sample_ms"],
            "full_test_set_ms": inference_metrics["full_test_set_ms"],
            "per_sample_avg_ms": inference_metrics["per_sample_avg_ms"],
            "test_set_size": inference_metrics["test_set_size"],
            "peak_ram_mb": inference_ram_mb,
        },
        "model_size": {
            "files": {k: round(v, 4) for k, v in model_sizes.items()},
            "total_mb": round(total_model_size_mb, 4),
        },
        "hardware": hardware,
    }

    # ─── Print Report ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("MODEL 1 — PERFORMANCE & HARDWARE REPORT")
    print(f"{'═' * 70}")

    print(f"\n{'─' * 70}")
    print("1. TRAINING")
    print(f"{'─' * 70}")
    print(f"  Training time:       {train_metrics['training_time_seconds']:.3f} seconds")
    print(f"  Peak RAM (training): {train_metrics['peak_memory_mb']:.3f} MB")
    print(f"  Train samples:       {X_train.shape[0]}")
    print(f"  Feature dimensions:  {X_train.shape[1]}")
    print(f"  Output classes:      {Y_train.shape[1]}")

    print(f"\n{'─' * 70}")
    print("2. INFERENCE TIME")
    print(f"{'─' * 70}")
    print(f"  Single sample:         {inference_metrics['single_sample_ms']:.4f} ms (avg over {inference_metrics['single_sample_repeats']} runs)")
    print(f"  Full test set ({inference_metrics['test_set_size']} samples): {inference_metrics['full_test_set_ms']:.4f} ms")
    print(f"  Per-sample average:    {inference_metrics['per_sample_avg_ms']:.4f} ms")
    print(f"  Peak RAM (inference):  {inference_ram_mb:.3f} MB")

    print(f"\n{'─' * 70}")
    print("3. MODEL SIZE ON DISK")
    print(f"{'─' * 70}")
    for fname, size in model_sizes.items():
        print(f"  {fname:<35} {size:.4f} MB")
    print(f"  {'─' * 45}")
    print(f"  {'TOTAL':<35} {total_model_size_mb:.4f} MB")

    print(f"\n{'─' * 70}")
    print("4. HARDWARE")
    print(f"{'─' * 70}")
    print(f"  Platform:       {hardware['platform']}")
    print(f"  Processor:      {hardware['processor']}")
    print(f"  Machine:        {hardware['machine']}")
    print(f"  Python:         {hardware['python_version']}")
    print(f"  CPU cores:      {hardware['cpu_count']}")
    print(f"  GPU available:  {hardware['gpu_available']}")

    print(f"\n{'═' * 70}\n")

    # ─── Save Report ──────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "model1_hardware_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
