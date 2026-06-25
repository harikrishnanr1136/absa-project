"""
Test Suite for Batch Processing Logic — without running Streamlit server.

Tests CSV validation, cleaning, result preparation, chart generation,
batch inference, and summary statistics.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.utils.csv_helpers import (
    validate_csv,
    clean_csv_for_inference,
    prepare_results_dataframe,
    results_to_download_csv,
    get_batch_summary_stats,
)
from app.utils.batch_runner import run_batch_inference, estimate_batch_time

# Track results
PASSED = 0
FAILED = 0
FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    """Record test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ PASSED — {name}")
    else:
        FAILED += 1
        msg = f"  ❌ FAILED — {name}"
        if detail:
            msg += f" [{detail}]"
        print(msg)
        FAILURES.append(f"{name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: CSV Validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_csv_validation():
    print(f"\n{'─' * 70}")
    print("TEST 1: CSV Validation")
    print(f"{'─' * 70}")

    # Valid CSV with "feedback" column
    df = pd.DataFrame({"feedback": ["good service", "bad network"]})
    result = validate_csv(df)
    check("Valid 'feedback' column → valid=True", result["valid"] is True)

    # Capital "Feedback"
    df = pd.DataFrame({"Feedback": ["test1", "test2"]})
    result = validate_csv(df)
    check("'Feedback' (capital) → valid=True", result["valid"] is True)
    check("'Feedback' column used correctly", result["column_used"] == "Feedback")

    # "text" column
    df = pd.DataFrame({"text": ["hello", "world"]})
    result = validate_csv(df)
    check("'text' column → valid=True", result["valid"] is True)
    check("'text' column renamed", result["column_used"] == "text")

    # No recognized column
    df = pd.DataFrame({"name": ["Alice"], "age": [30]})
    result = validate_csv(df)
    check("No recognized column → valid=False", result["valid"] is False)
    check("Error message mentions columns", "column" in result["error"].lower())

    # Empty CSV
    df = pd.DataFrame()
    result = validate_csv(df)
    check("Empty CSV → valid=False", result["valid"] is False)

    # CSV with 1001 rows
    df = pd.DataFrame({"feedback": [f"row {i}" for i in range(1001)]})
    result = validate_csv(df)
    check("1001 rows → valid=False", result["valid"] is False)
    check("Error mentions limit", "1000" in str(result["error"]))

    # All null feedbacks
    df = pd.DataFrame({"feedback": [None, None, None]})
    result = validate_csv(df)
    check("All null feedbacks → valid=False", result["valid"] is False)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: clean_csv_for_inference
# ═══════════════════════════════════════════════════════════════════════════════

def test_clean_csv():
    print(f"\n{'─' * 70}")
    print("TEST 2: clean_csv_for_inference")
    print(f"{'─' * 70}")

    # 10 rows: 3 nulls, 2 empty strings → 5 valid
    df = pd.DataFrame({"feedback": [
        "Good service",       # 1 - valid
        None,                 # 2 - null
        "Bad network",        # 3 - valid
        "",                   # 4 - empty
        "OK speed",           # 5 - valid
        None,                 # 6 - null
        "   ",                # 7 - whitespace only → empty after strip
        "Fast 5G",            # 8 - valid
        None,                 # 9 - null
        "Great value",        # 10 - valid
    ]})

    cleaned = clean_csv_for_inference(df)

    check("5 valid rows returned", len(cleaned) == 5, detail=f"got {len(cleaned)}")
    check("Has 'original_row_number' column", "original_row_number" in cleaned.columns)
    check("Has 'feedback' column", "feedback" in cleaned.columns)

    # Verify original row numbers preserved
    expected_rows = [1, 3, 5, 8, 10]
    actual_rows = cleaned["original_row_number"].tolist()
    check("Original row numbers preserved", actual_rows == expected_rows,
          detail=f"expected {expected_rows}, got {actual_rows}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: prepare_results_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

def test_prepare_results():
    print(f"\n{'─' * 70}")
    print("TEST 3: prepare_results_dataframe")
    print(f"{'─' * 70}")

    cleaned_df = pd.DataFrame({
        "original_row_number": [1, 2, 3, 4, 5],
        "feedback": ["a", "b", "c", "d", "e"],
    })

    predictions = [
        {"feedback": "a", "detected_aspects": ["network_coverage"],
         "aspect_sentiments": {"network_coverage": "positive"},
         "confidence_scores": {"network_coverage": 0.9},
         "overall_sentiment": "positive", "inference_time_ms": 100},
        {"feedback": "b", "detected_aspects": ["internet_speed", "billing"],
         "aspect_sentiments": {"internet_speed": "negative", "billing": "negative"},
         "confidence_scores": {"internet_speed": 0.8, "billing": 0.7},
         "overall_sentiment": "negative", "inference_time_ms": 120},
        {"feedback": "c", "detected_aspects": [],
         "aspect_sentiments": {},
         "confidence_scores": {},
         "overall_sentiment": "neutral", "inference_time_ms": 80},
        {"feedback": "d", "detected_aspects": ["5g_experience"],
         "aspect_sentiments": {"5g_experience": "positive"},
         "confidence_scores": {"5g_experience": 0.85},
         "overall_sentiment": "positive", "inference_time_ms": 110},
        {"feedback": "e", "detected_aspects": [],
         "aspect_sentiments": {},
         "confidence_scores": {},
         "overall_sentiment": "error", "inference_time_ms": 0, "error": "test"},
    ]

    results_df = prepare_results_dataframe(cleaned_df, predictions)

    # All 8 required columns
    required_cols = ["original_row_number", "feedback", "detected_aspects",
                     "aspect_sentiments", "confidence_scores", "overall_sentiment",
                     "inference_time_ms", "aspect_count"]
    for col in required_cols:
        check(f"Has column '{col}'", col in results_df.columns)

    check("5 rows in output", len(results_df) == 5)

    # detected_aspects is comma-separated string
    row2 = results_df[results_df["original_row_number"] == 2].iloc[0]
    check("detected_aspects is comma-separated",
          "internet_speed" in row2["detected_aspects"] and "billing" in row2["detected_aspects"])

    # aspect_sentiments is valid JSON
    try:
        parsed = json.loads(row2["aspect_sentiments"])
        check("aspect_sentiments is valid JSON", isinstance(parsed, dict))
    except json.JSONDecodeError:
        check("aspect_sentiments is valid JSON", False, detail="JSONDecodeError")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Dashboard Charts
# ═══════════════════════════════════════════════════════════════════════════════

def test_dashboard_charts():
    print(f"\n{'─' * 70}")
    print("TEST 4: Dashboard Charts (all 7 return Figure, no exceptions)")
    print(f"{'─' * 70}")

    from app.utils.dashboard_charts import (
        sentiment_distribution_pie,
        aspect_frequency_bar,
        positive_negative_trend_bar,
        aspect_sentiment_heatmap,
        confidence_distribution_histogram,
        top_negative_aspects_bar,
        feedback_length_vs_aspect_count_scatter,
    )
    import plotly.graph_objects as go

    # Mock results with 20 rows
    rows = []
    aspects_pool = ["network_coverage", "internet_speed", "billing", "customer_support", "5g_experience"]
    sentiments_pool = ["positive", "negative", "neutral"]
    for i in range(20):
        aspect = aspects_pool[i % len(aspects_pool)]
        sent = sentiments_pool[i % len(sentiments_pool)]
        rows.append({
            "feedback": f"Test feedback number {i} with some words here",
            "detected_aspects": aspect,
            "aspect_sentiments": json.dumps({aspect: sent}),
            "confidence_scores": json.dumps({aspect: 0.5 + (i % 5) * 0.1}),
            "overall_sentiment": sent,
            "inference_time_ms": 100 + i * 5,
            "aspect_count": 1,
        })
    mock_df = pd.DataFrame(rows)

    chart_funcs = [
        ("sentiment_distribution_pie", sentiment_distribution_pie),
        ("aspect_frequency_bar", aspect_frequency_bar),
        ("positive_negative_trend_bar", positive_negative_trend_bar),
        ("aspect_sentiment_heatmap", aspect_sentiment_heatmap),
        ("confidence_distribution_histogram", confidence_distribution_histogram),
        ("top_negative_aspects_bar", top_negative_aspects_bar),
        ("feedback_length_vs_aspect_count_scatter", feedback_length_vs_aspect_count_scatter),
    ]

    for name, func in chart_funcs:
        try:
            fig = func(mock_df)
            check(f"{name} returns Figure", isinstance(fig, go.Figure))
        except Exception as e:
            check(f"{name} no exception", False, detail=str(e))

    # Edge case: all no aspects
    empty_aspects_df = pd.DataFrame([{
        "feedback": "test", "detected_aspects": "", "aspect_sentiments": "{}",
        "confidence_scores": "{}", "overall_sentiment": "neutral",
        "inference_time_ms": 50, "aspect_count": 0,
    }] * 5)

    try:
        fig = aspect_frequency_bar(empty_aspects_df)
        check("Empty aspects edge case → no crash", isinstance(fig, go.Figure))
    except Exception as e:
        check("Empty aspects edge case", False, detail=str(e))

    # Edge case: all same sentiment
    all_pos_df = mock_df.copy()
    all_pos_df["overall_sentiment"] = "positive"
    try:
        fig = sentiment_distribution_pie(all_pos_df)
        check("All-positive edge case → no crash", isinstance(fig, go.Figure))
    except Exception as e:
        check("All-positive edge case", False, detail=str(e))

    # Edge case: single row
    single_df = mock_df.head(1)
    try:
        fig = feedback_length_vs_aspect_count_scatter(single_df)
        check("Single row edge case → no crash", isinstance(fig, go.Figure))
    except Exception as e:
        check("Single row edge case", False, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Full Batch Inference
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_inference():
    print(f"\n{'─' * 70}")
    print("TEST 5: Batch Inference (mock pipeline)")
    print(f"{'─' * 70}")

    # Mock pipeline
    class MockPipeline:
        def predict(self, text):
            if "ERROR_TRIGGER" in text:
                raise ValueError("Simulated error")
            return {
                "feedback": text,
                "detected_aspects": ["network_coverage"],
                "aspect_sentiments": {"network_coverage": "positive"},
                "confidence_scores": {"network_coverage": 0.88},
                "overall_sentiment": "positive",
                "inference_time_ms": 50.0,
            }

    # Mock Streamlit elements
    class MockProgress:
        def progress(self, v): pass

    class MockStatus:
        def text(self, msg): pass

    test_feedbacks = [
        "Great network coverage.",
        "Internet speed is terrible.",
        "ERROR_TRIGGER this should fail gracefully.",
        "Customer support was helpful.",
        "5G is fast.",
        "Billing issues again.",
        "Recharge plan is good value.",
        "ERROR_TRIGGER another failure.",
        "App works smoothly.",
        "SIM activation was quick.",
    ]

    results = run_batch_inference(MockPipeline(), test_feedbacks, MockProgress(), MockStatus())

    check("Output length == 10", len(results) == 10, detail=f"got {len(results)}")
    check("No None in results", all(r is not None for r in results))

    # Error rows
    error_rows = [r for r in results if r["overall_sentiment"] == "error"]
    check("2 error rows detected", len(error_rows) == 2, detail=f"got {len(error_rows)}")
    check("Error rows have 'error' key", all("error" in r for r in error_rows))

    # Success rows
    success_rows = [r for r in results if r["overall_sentiment"] != "error"]
    check("8 success rows", len(success_rows) == 8)
    check("Success rows have detected_aspects", all("detected_aspects" in r for r in success_rows))

    # Estimate time
    est = estimate_batch_time(100, 500)
    check("estimate_batch_time returns string", isinstance(est, str))
    check("estimate contains time unit", "second" in est or "minute" in est)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: get_batch_summary_stats
# ═══════════════════════════════════════════════════════════════════════════════

def test_summary_stats():
    print(f"\n{'─' * 70}")
    print("TEST 6: get_batch_summary_stats")
    print(f"{'─' * 70}")

    # Mock results with known distributions: 4 positive, 3 negative, 3 neutral = 10
    rows = []
    sentiments = ["positive"] * 4 + ["negative"] * 3 + ["neutral"] * 3
    aspects = ["network_coverage", "internet_speed", "billing", "customer_support",
               "network_coverage", "internet_speed", "billing",
               "network_coverage", "internet_speed", "5g_experience"]

    for i in range(10):
        aspect = aspects[i]
        sent = sentiments[i]
        rows.append({
            "original_row_number": i + 1,
            "feedback": f"Feedback {i}",
            "detected_aspects": aspect,
            "aspect_sentiments": json.dumps({aspect: sent}),
            "confidence_scores": json.dumps({aspect: 0.8}),
            "overall_sentiment": sent,
            "inference_time_ms": 100.0,
            "aspect_count": 1,
        })

    results_df = pd.DataFrame(rows)
    stats = get_batch_summary_stats(results_df)

    # All required keys
    required_keys = ["total_rows", "rows_with_aspects", "rows_without_aspects",
                     "sentiment_distribution", "most_common_aspect",
                     "most_common_positive_aspect", "most_common_negative_aspect",
                     "avg_aspects_per_feedback", "avg_inference_time_ms"]

    for key in required_keys:
        check(f"Has key '{key}'", key in stats, detail=f"keys={list(stats.keys())}")

    check("total_rows == 10", stats["total_rows"] == 10)
    check("rows_with_aspects == 10", stats["rows_with_aspects"] == 10)
    check("rows_without_aspects == 0", stats["rows_without_aspects"] == 0)
    check("most_common_aspect is network_coverage",
          stats["most_common_aspect"] == "network_coverage",
          detail=f"got {stats['most_common_aspect']}")

    # Percentages sum to ~100
    pcts = stats["sentiment_distribution"]["percentages"]
    total_pct = sum(pcts.values())
    check("Percentages sum to ~100", abs(total_pct - 100.0) < 1.0,
          detail=f"got {total_pct}")

    # Verify counts
    counts = stats["sentiment_distribution"]["counts"]
    check("Positive count == 4", counts.get("positive") == 4)
    check("Negative count == 3", counts.get("negative") == 3)
    check("Neutral count == 3", counts.get("neutral") == 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("BATCH PROCESSING — TEST SUITE")
    print("(Tests logic without running Streamlit server)")
    print("=" * 70)

    test_csv_validation()
    test_clean_csv()
    test_prepare_results()
    test_dashboard_charts()
    test_batch_inference()
    test_summary_stats()

    # Summary
    print(f"\n{'═' * 70}")
    print("TEST SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  Total: {PASSED + FAILED}")
    print(f"  Passed: {PASSED}")
    print(f"  Failed: {FAILED}")

    if FAILURES:
        print(f"\n  FAILURES:")
        for f in FAILURES:
            print(f"    • {f}")

    if FAILED == 0:
        print(f"\n  🎉 ALL TESTS PASSED — Batch processing logic verified")
    else:
        print(f"\n  ⚠️  {FAILED} test(s) failed")

    print(f"{'═' * 70}\n")

    assert FAILED == 0, f"{FAILED} test(s) failed"


if __name__ == "__main__":
    main()
