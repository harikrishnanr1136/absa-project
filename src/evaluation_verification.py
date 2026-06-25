"""
Evaluation Artifact Verification Script.

Verifies all Day 4 and Day 5 evaluation artifacts are complete
before starting Day 6 analysis.

Exits with code 1 if any check fails.
"""

import json
import logging
import os
import sys

import pandas as pd

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Track overall pass/fail
all_passed = True
failures = []


def check(name: str, condition: bool, detail: str = ""):
    """Record a check result."""
    global all_passed
    if condition:
        logger.info(f"  ✅ PASSED — {name}")
    else:
        all_passed = False
        msg = f"  ❌ FAILED — {name}"
        if detail:
            msg += f" [{detail}]"
        logger.error(msg)
        failures.append(f"{name}: {detail}")


def file_exists(path: str, label: str) -> bool:
    """Check file exists and log result."""
    exists = os.path.exists(path)
    check(f"{label} exists", exists, detail=f"Missing: {path}" if not exists else "")
    return exists


def load_json_checked(path: str, label: str) -> dict:
    """Load JSON file with existence check."""
    if not file_exists(path, label):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        check(f"{label} valid JSON", False, detail=str(e))
        return {}


def verify_metrics_structure(data: dict, label: str, require_confusion_matrix: bool = False):
    """
    Verify model metrics file has required structure:
    - aspect_detection metrics for train, val, test
    - sentiment_classification metrics for train, val, test
    """
    # Check top-level keys
    for section in ["aspect_detection", "sentiment_classification"]:
        has_section = section in data
        check(f"{label} has '{section}'", has_section,
              detail=f"Missing key: {section}" if not has_section else "")

        if not has_section:
            continue

        section_data = data[section]

        # Check splits exist
        for split in ["train", "val", "test"]:
            has_split = split in section_data
            check(f"{label}.{section} has '{split}'", has_split,
                  detail=f"Missing split: {split}" if not has_split else "")

            if not has_split:
                continue

            split_data = section_data[split]

            # Check required metric keys based on section type
            if section == "aspect_detection":
                # Can be nested (micro/macro) or flat
                if isinstance(split_data, dict):
                    has_f1 = ("micro" in split_data or "micro_f1" in split_data or
                              "f1" in split_data or "macro" in split_data or "macro_f1" in split_data)
                    check(f"{label}.{section}.{split} has F1 metrics", has_f1,
                          detail=f"Keys found: {list(split_data.keys())}" if not has_f1 else "")

            elif section == "sentiment_classification":
                # Check for overall or per-aspect metrics
                if isinstance(split_data, dict):
                    has_metrics = ("_overall" in split_data or "macro_f1" in split_data or
                                   "accuracy" in split_data)
                    check(f"{label}.{section}.{split} has metrics", has_metrics,
                          detail=f"Keys found: {list(split_data.keys())[:5]}" if not has_metrics else "")


def verify_hardware_report(data: dict, label: str):
    """Verify hardware report has required keys."""
    # Flatten all keys recursively for flexible matching
    def flatten_keys(d, prefix=""):
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            if isinstance(v, dict):
                keys.extend(flatten_keys(v, full_key))
        return keys

    all_keys = " ".join(flatten_keys(data)).lower()

    required_concepts = {
        "training_time": ["time_seconds", "training_time", "train_time", "training"],
        "inference_time": ["per_sample_avg_ms", "inference_time", "single_sample_ms", "peak_inference_memory"],
        "peak_memory": ["peak_ram_mb", "peak_memory", "ram_mb", "peak_inference_memory_mb"],
        "model_size": ["model_size", "total_mb", "gpu_memory_mb"],
    }

    for concept, variants in required_concepts.items():
        found = any(v in all_keys for v in variants)
        check(f"{label} contains '{concept}' info", found,
              detail=f"Could not find any of {variants} in keys" if not found else "")


def main():
    global all_passed

    logger.info("=" * 70)
    logger.info("EVALUATION ARTIFACT VERIFICATION")
    logger.info("Day 4 + Day 5 completeness check before Day 6 analysis")
    logger.info("=" * 70)

    config = load_config()
    output_dir = resolve_path(config["outputs"]["models"])
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))

    # ─── Check 1: Model 1 Metrics ─────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 1: Model 1 Metrics")
    logger.info("─" * 70)

    m1_path = os.path.join(output_dir, "model1_metrics.json")

    # Model 1 metrics live in experiment_log.json
    m1_data = {}
    if os.path.exists(m1_path):
        m1_data = load_json_checked(m1_path, "model1_metrics.json")
    else:
        # Fallback: load from experiment_log.json
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for candidate in [
            os.path.join(project_root, "outputs", "experiment_log.json"),
            os.path.join(output_dir, "experiment_log.json"),
        ]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    exp_log = json.load(f)
                m1_entries = [e for e in exp_log if e.get("experiment_id") == "model1_lr_tfidf"]
                if m1_entries:
                    m1_data = m1_entries[0]
                    check("Model 1 metrics found (via experiment_log.json)", True)
                    break
        if not m1_data:
            check("Model 1 metrics found", False,
                  detail=f"Not in {m1_path} or experiment_log.json")

    if m1_data:
        verify_metrics_structure(m1_data, "Model 1")

    # ─── Check 2: Model 2 Metrics ─────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 2: Model 2 Metrics (outputs/models/model2_metrics.json)")
    logger.info("─" * 70)

    m2_path = os.path.join(output_dir, "model2_metrics.json")
    m2_data = load_json_checked(m2_path, "model2_metrics.json")

    if not m2_data:
        # Check experiment log fallback
        exp_log_path = os.path.join(os.path.dirname(output_dir), "experiment_log.json")
        if os.path.exists(exp_log_path):
            with open(exp_log_path) as f:
                exp_log = json.load(f)
            m2_entries = [e for e in exp_log if e.get("experiment_id") == "model2_distilbert_finetuned"]
            if m2_entries:
                m2_data = m2_entries[0]
                logger.info("  (Loaded Model 2 metrics from experiment_log.json)")

    if m2_data:
        verify_metrics_structure(m2_data, "Model 2")

    # ─── Check 3: Model 1 Hardware Report ─────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 3: Model 1 Hardware Report")
    logger.info("─" * 70)

    m1_hw_path = os.path.join(output_dir, "model1_hardware_report.json")
    m1_hw = load_json_checked(m1_hw_path, "model1_hardware_report.json")

    if m1_hw:
        verify_hardware_report(m1_hw, "Model 1 HW")

    # ─── Check 4: Model 2 Hardware Report ─────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 4: Model 2 Hardware Report")
    logger.info("─" * 70)

    m2_hw_path = os.path.join(output_dir, "model2_hardware_report.json")
    m2_hw = load_json_checked(m2_hw_path, "model2_hardware_report.json")

    if m2_hw:
        verify_hardware_report(m2_hw, "Model 2 HW")

    # ─── Check 5: Experiment Log ──────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 5: Experiment Log (exactly 2 entries)")
    logger.info("─" * 70)

    # experiment_log.json lives at outputs/ not outputs/models/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_log_candidates = [
        os.path.join(project_root, "outputs", "experiment_log.json"),
        os.path.join(output_dir, "experiment_log.json"),
    ]
    exp_log_path = None
    for candidate in exp_log_candidates:
        if os.path.exists(candidate):
            exp_log_path = candidate
            break
    if exp_log_path is None:
        exp_log_path = exp_log_candidates[0]  # Use first for error message
    exp_log = load_json_checked(exp_log_path, "experiment_log.json")

    if exp_log:
        check("experiment_log is a list", isinstance(exp_log, list),
              detail=f"Got type: {type(exp_log).__name__}")

        if isinstance(exp_log, list):
            check("experiment_log has exactly 2 entries", len(exp_log) == 2,
                  detail=f"Found {len(exp_log)} entries, expected 2")

            ids = [e.get("experiment_id", "?") for e in exp_log]
            check("Contains model1_lr_tfidf", "model1_lr_tfidf" in ids,
                  detail=f"IDs found: {ids}")
            check("Contains model2_distilbert_finetuned", "model2_distilbert_finetuned" in ids,
                  detail=f"IDs found: {ids}")

    # ─── Check 6: Test Data ───────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CHECK 6: Test data (data/test.csv)")
    logger.info("─" * 70)

    test_path = os.path.join(data_dir, "test.csv")
    if file_exists(test_path, "data/test.csv"):
        try:
            test_df = pd.read_csv(test_path)

            required_cols = ["feedback", "aspects", "aspect_sentiments"]
            for col in required_cols:
                check(f"test.csv has column '{col}'", col in test_df.columns,
                      detail=f"Columns: {list(test_df.columns)}" if col not in test_df.columns else "")

            check("test.csv has >= 100 rows", len(test_df) >= 100,
                  detail=f"Found {len(test_df)} rows")

        except Exception as e:
            check("test.csv readable", False, detail=str(e))

    # ─── Final Summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)

    if all_passed:
        logger.info("🎉 ALL VERIFICATION CHECKS PASSED")
        logger.info("   Ready to proceed with Day 6 analysis.")
    else:
        logger.error(f"⚠️  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            logger.error(f"   • {f}")
        logger.error("")
        logger.error("Fix the above issues before proceeding.")

    logger.info("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
