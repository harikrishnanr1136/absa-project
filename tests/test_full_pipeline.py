"""
Master Integration Test — Full Pipeline End-to-End Verification.

Tests config, preprocessing, feature engineering, model loading, inference,
batch processing, CSV utilities, and dashboard charts as a unified system.

Saves report to outputs/integration_test_report.txt.
"""

import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── Test Framework ───────────────────────────────────────────────────────────
PASSED = 0
FAILED = 0
FAILURES = []
REPORT_LINES = []


def check(name: str, condition: bool, detail: str = ""):
    """Record a test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        line = f"  ✅ PASSED — {name}"
    else:
        FAILED += 1
        line = f"  ❌ FAILED — {name}"
        if detail:
            line += f" [{detail}]"
        FAILURES.append(f"{name}: {detail}")
    print(line)
    REPORT_LINES.append(line)


def section(title: str):
    """Print and record section header."""
    header = f"\n{'─' * 70}\n{title}\n{'─' * 70}"
    print(header)
    REPORT_LINES.append(header)


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 1: Config
# ═══════════════════════════════════════════════════════════════════════════════

def test_config():
    section("SUITE 1 — Config")

    try:
        from src.config import load_config, resolve_path
        config = load_config()
        check("load_config() succeeds", True)
    except Exception as e:
        check("load_config() succeeds", False, detail=str(e))
        return

    # 15 aspects
    aspects = config.get("labels", {}).get("aspects", [])
    check("15 aspects in config", len(aspects) == 15, detail=f"got {len(aspects)}")

    # 3 sentiments
    sentiments = config.get("labels", {}).get("sentiments", [])
    check("3 sentiments in config", len(sentiments) == 3, detail=f"got {sentiments}")

    # Paths resolve
    for key in ["cleaned", "raw"]:
        path = resolve_path(config["data"][key])
        check(f"data.{key} path resolves", isinstance(path, str) and len(path) > 0)

    # Models dir
    models_dir = config.get("models", {}).get("dir", "")
    check("models.dir configured", len(models_dir) > 0, detail=f"got '{models_dir}'")


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 2: Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def test_preprocessing():
    section("SUITE 2 — Preprocessing")

    try:
        from src.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline()
        check("PreprocessingPipeline loads", True)
    except Exception as e:
        check("PreprocessingPipeline loads", False, detail=str(e))
        return

    samples = [
        "The 5G speed is amazing but customer support is terrible.",
        "plz fix ur network coverage. cant even make calls.",
        "SIM activation was quick. Happy with the service.",
        "OTT bundle with Netflix is great value for money.",
        "Billing has hidden charges every month.",
        "Internet speed drops to nothing after 8pm daily.",
        "Roaming charges are way too high.",
        "The mobile app crashes during recharge.",
        "Call quality is crystal clear on VoLTE.",
        "Data balance drains too fast overnight.",
    ]

    results = pipeline.fit_transform(samples)
    check("fit_transform output length matches", len(results) == 10)
    check("No empty strings in output", all(len(r.strip()) > 0 for r in results))

    # Domain terms preserved
    all_text = " ".join(results)
    check("'5g' preserved", "5g" in all_text)
    check("'sim' preserved", "sim" in all_text)
    check("'ott' preserved", "ott" in all_text)


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 3: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def test_features():
    section("SUITE 3 — Feature Engineering")

    from src.preprocessing import PreprocessingPipeline
    from src.features import TFIDFFeatures

    pipeline = PreprocessingPipeline()
    texts = pipeline.fit_transform(["good network", "bad speed", "ok billing"] * 5)

    # TF-IDF
    tfidf = TFIDFFeatures()
    X = tfidf.fit_transform(texts)
    check("TF-IDF shape rows match", X.shape[0] == 15)
    check("TF-IDF features <= 10000", X.shape[1] <= 10000)
    check("TF-IDF no NaN", not np.isnan(X.toarray()).any())
    check("TF-IDF all values >= 0", X.toarray().min() >= 0)

    # Sentence Embeddings (optional — may not be available)
    try:
        from src.features import SentenceEmbeddingFeatures
        embedder = SentenceEmbeddingFeatures()
        X_emb = embedder.fit_transform(["test sentence"])
        check("Embeddings shape (n, 384)", X_emb.shape[1] == 384)
        check("Embeddings no NaN", not np.isnan(X_emb).any())
        check("Embeddings no Inf", not np.isinf(X_emb).any())
    except ImportError:
        check("SentenceEmbeddings (skipped — not installed)", True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 4: Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_loading():
    section("SUITE 4 — Model Loading")

    try:
        from src.inference import ABSAInferencePipeline
        pipeline = ABSAInferencePipeline()
        check("ABSAInferencePipeline loads", True)
    except Exception as e:
        check("ABSAInferencePipeline loads", False, detail=str(e))
        return

    check("Has tokenizer", pipeline.tokenizer is not None)
    check("Has aspect_model", pipeline.aspect_model is not None)
    check("Has sentiment_models dict", isinstance(pipeline.sentiment_models, dict))
    check("15 sentiment models loaded", len(pipeline.sentiment_models) == 15,
          detail=f"got {len(pipeline.sentiment_models)}")

    return pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 5: Inference
# ═══════════════════════════════════════════════════════════════════════════════

def test_inference(pipeline):
    section("SUITE 5 — Inference")

    if pipeline is None:
        check("Skipped (pipeline not loaded)", True)
        return

    # Standard feedbacks
    standard = [
        "Internet speed is excellent but billing is confusing.",
        "Network coverage is poor in my area.",
        "Customer support was helpful.",
        "5G experience is amazing.",
        "Recharge plans are too expensive.",
    ]

    for i, text in enumerate(standard, 1):
        result = pipeline.predict(text)
        check(f"Standard {i}: returns dict", isinstance(result, dict))
        check(f"Standard {i}: has detected_aspects", "detected_aspects" in result)
        check(f"Standard {i}: aspects match sentiments",
              set(result.get("aspect_sentiments", {}).keys()) == set(result.get("detected_aspects", [])))
        check(f"Standard {i}: inference_time_ms > 0", result.get("inference_time_ms", 0) > 0)

    # Edge cases
    edges = [("empty", ""), ("short", "bad"), ("noisy", "plz fix ur network"),
             ("upper", "VERY BAD SERVICE"), ("repeated", "good " * 20)]

    for label, text in edges:
        try:
            result = pipeline.predict(text)
            check(f"Edge '{label}': no exception", True)
            # Confidence scores in [0, 1]
            for score in result.get("confidence_scores", {}).values():
                if not (0 <= score <= 1):
                    check(f"Edge '{label}': confidence in [0,1]", False, detail=f"got {score}")
                    break
            else:
                check(f"Edge '{label}': confidence in [0,1]", True)
        except Exception as e:
            check(f"Edge '{label}': no exception", False, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 6: Batch Inference
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_inference(pipeline):
    section("SUITE 6 — Batch Inference")

    if pipeline is None:
        check("Skipped (pipeline not loaded)", True)
        return

    from app.utils.batch_runner import run_batch_inference

    class MockProgress:
        def progress(self, v): pass
    class MockStatus:
        def text(self, msg): pass

    feedbacks = [f"Test feedback number {i} about network and billing" for i in range(20)]

    start = time.time()
    results = run_batch_inference(pipeline, feedbacks, MockProgress(), MockStatus())
    elapsed = time.time() - start

    check("Batch output length == 20", len(results) == 20)
    check("No None in results", all(r is not None for r in results))
    check("All results are dicts", all(isinstance(r, dict) for r in results))
    check(f"Batch completed in {elapsed:.1f}s", elapsed < 300)  # Under 5 min


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 7: CSV Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def test_csv_utilities():
    section("SUITE 7 — CSV Utilities")

    from app.utils.csv_helpers import (
        validate_csv, clean_csv_for_inference,
        prepare_results_dataframe, results_to_download_csv,
    )

    # validate_csv
    check("Valid CSV", validate_csv(pd.DataFrame({"feedback": ["a", "b"]}))["valid"])
    check("Capital Feedback", validate_csv(pd.DataFrame({"Feedback": ["a"]}))["valid"])
    check("'text' column", validate_csv(pd.DataFrame({"text": ["a"]}))["valid"])
    check("No column → invalid", not validate_csv(pd.DataFrame({"x": [1]}))["valid"])
    check("Empty → invalid", not validate_csv(pd.DataFrame())["valid"])
    check("1001 rows → invalid",
          not validate_csv(pd.DataFrame({"feedback": ["x"] * 1001}))["valid"])

    # clean_csv_for_inference
    df = pd.DataFrame({"feedback": ["a", None, "b", "", "c"]})
    cleaned = clean_csv_for_inference(df)
    check("clean_csv: 3 valid rows", len(cleaned) == 3)
    check("clean_csv: has original_row_number", "original_row_number" in cleaned.columns)

    # prepare_results_dataframe
    preds = [{"feedback": "a", "detected_aspects": ["x"], "aspect_sentiments": {"x": "pos"},
              "confidence_scores": {"x": 0.9}, "overall_sentiment": "positive",
              "inference_time_ms": 50}]
    results_df = prepare_results_dataframe(cleaned.head(1), preds)
    check("prepare_results: has aspect_count", "aspect_count" in results_df.columns)

    # results_to_download_csv
    csv_bytes = results_to_download_csv(results_df)
    check("download_csv returns bytes", isinstance(csv_bytes, bytes))
    check("download_csv not empty", len(csv_bytes) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 8: Dashboard Charts
# ═══════════════════════════════════════════════════════════════════════════════

def test_dashboard_charts():
    section("SUITE 8 — Dashboard Charts")

    from app.utils.dashboard_charts import (
        sentiment_distribution_pie, aspect_frequency_bar,
        positive_negative_trend_bar, aspect_sentiment_heatmap,
        confidence_distribution_histogram, top_negative_aspects_bar,
        feedback_length_vs_aspect_count_scatter,
    )
    import plotly.graph_objects as go

    # Mock data
    mock = pd.DataFrame([{
        "feedback": f"Feedback {i}",
        "detected_aspects": "network_coverage, billing",
        "aspect_sentiments": json.dumps({"network_coverage": "positive", "billing": "negative"}),
        "confidence_scores": json.dumps({"network_coverage": 0.9, "billing": 0.7}),
        "overall_sentiment": ["positive", "negative", "neutral"][i % 3],
        "inference_time_ms": 100 + i, "aspect_count": 2,
    } for i in range(15)])

    charts = [
        ("sentiment_distribution_pie", sentiment_distribution_pie),
        ("aspect_frequency_bar", aspect_frequency_bar),
        ("positive_negative_trend_bar", positive_negative_trend_bar),
        ("aspect_sentiment_heatmap", aspect_sentiment_heatmap),
        ("confidence_distribution_histogram", confidence_distribution_histogram),
        ("top_negative_aspects_bar", top_negative_aspects_bar),
        ("feedback_length_vs_aspect_count_scatter", feedback_length_vs_aspect_count_scatter),
    ]

    for name, func in charts:
        try:
            fig = func(mock)
            check(f"{name} returns Figure", isinstance(fig, go.Figure))
        except Exception as e:
            check(f"{name} no exception", False, detail=str(e))

    # Edge case: empty aspects
    empty_df = pd.DataFrame([{"feedback": "x", "detected_aspects": "",
                              "aspect_sentiments": "{}", "confidence_scores": "{}",
                              "overall_sentiment": "neutral", "inference_time_ms": 50,
                              "aspect_count": 0}])
    try:
        fig = aspect_frequency_bar(empty_df)
        check("Empty aspects edge case", isinstance(fig, go.Figure))
    except Exception as e:
        check("Empty aspects edge case", False, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    header = "=" * 70 + "\nMASTER INTEGRATION TEST — Full Pipeline\n" + "=" * 70
    print(header)
    REPORT_LINES.append(header)

    # Run suites
    test_config()
    test_preprocessing()
    test_features()
    pipeline = test_model_loading()
    test_inference(pipeline)
    test_batch_inference(pipeline)
    test_csv_utilities()
    test_dashboard_charts()

    # Summary
    summary = f"""
{'═' * 70}
INTEGRATION TEST SUMMARY
{'═' * 70}

  Total tests:  {PASSED + FAILED}
  Passed:       {PASSED}
  Failed:       {FAILED}
"""
    if FAILURES:
        summary += f"\n  FAILURES:\n"
        for f in FAILURES:
            summary += f"    • {f}\n"

    if FAILED == 0:
        summary += f"\n  🎉 ALL TESTS PASSED — System is production-ready\n"
    else:
        summary += f"\n  ⚠️  {FAILED} test(s) failed — fix before deploying\n"

    summary += "═" * 70
    print(summary)
    REPORT_LINES.append(summary)

    # Save report
    report_path = os.path.join(PROJECT_ROOT, "outputs", "integration_test_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(REPORT_LINES))
    print(f"\nReport saved: {report_path}")

    assert FAILED == 0, f"{FAILED} integration test(s) failed"


if __name__ == "__main__":
    main()
