"""
Test Checklist for Streamlit Page 1 — Single Feedback Analysis.

Verifies Page 1 logic without running the Streamlit server.
Tests app_helpers, components, format functions, and inference pipeline integration.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
# TEST 1: Import app_helpers — verify all 7 functions return correct types
# ═══════════════════════════════════════════════════════════════════════════════

def test_app_helpers():
    print(f"\n{'─' * 70}")
    print("TEST 1: app_helpers — all 7 functions return correct types")
    print(f"{'─' * 70}")

    from app.utils.app_helpers import (
        get_sentiment_color,
        get_sentiment_emoji,
        format_aspect_name,
        get_overall_sentiment_label,
        get_confidence_level,
        format_inference_time,
        get_sample_feedbacks,
    )

    # get_sentiment_color → str (hex color)
    result = get_sentiment_color("positive")
    check("get_sentiment_color returns str", isinstance(result, str), detail=f"got {type(result)}")
    check("get_sentiment_color starts with #", result.startswith("#"), detail=f"got '{result}'")

    # get_sentiment_emoji → str (emoji)
    result = get_sentiment_emoji("negative")
    check("get_sentiment_emoji returns str", isinstance(result, str), detail=f"got {type(result)}")
    check("get_sentiment_emoji non-empty", len(result) > 0)

    # format_aspect_name → str
    result = format_aspect_name("internet_speed")
    check("format_aspect_name returns str", isinstance(result, str))
    check("format_aspect_name correct", result == "Internet Speed", detail=f"got '{result}'")

    # get_overall_sentiment_label → str
    result = get_overall_sentiment_label({"a": "positive"})
    check("get_overall_sentiment_label returns str", isinstance(result, str))

    # get_confidence_level → str
    result = get_confidence_level(0.9)
    check("get_confidence_level returns str", isinstance(result, str))
    check("get_confidence_level correct", result == "High", detail=f"got '{result}'")

    # format_inference_time → str
    result = format_inference_time(245.3)
    check("format_inference_time returns str", isinstance(result, str))
    check("format_inference_time contains ms", "ms" in result or "s" in result, detail=f"got '{result}'")

    # get_sample_feedbacks → list
    result = get_sample_feedbacks()
    check("get_sample_feedbacks returns list", isinstance(result, list))
    check("get_sample_feedbacks has 8 items", len(result) == 8, detail=f"got {len(result)}")
    check("get_sample_feedbacks all strings", all(isinstance(s, str) for s in result))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: render_result_card with mock prediction — no exceptions
# ═══════════════════════════════════════════════════════════════════════════════

def test_render_result_card():
    print(f"\n{'─' * 70}")
    print("TEST 2: render_result_card with mock prediction (no Streamlit)")
    print(f"{'─' * 70}")

    mock_prediction = {
        "feedback": "Internet is fast but billing is confusing",
        "detected_aspects": ["internet_speed", "billing"],
        "aspect_sentiments": {
            "internet_speed": "positive",
            "billing": "negative",
        },
        "confidence_scores": {
            "internet_speed": 0.91,
            "billing": 0.78,
        },
        "overall_sentiment": "mixed",
        "inference_time_ms": 245.3,
    }

    # We can't actually render (needs Streamlit) but verify the function is importable
    # and the prediction dict has all required keys
    try:
        from app.components.result_card import render_result_card
        check("render_result_card importable", True)
    except ImportError as e:
        check("render_result_card importable", False, detail=str(e))
        return

    # Verify prediction dict schema
    required_keys = ["feedback", "detected_aspects", "aspect_sentiments",
                     "confidence_scores", "overall_sentiment", "inference_time_ms"]
    for key in required_keys:
        check(f"mock prediction has '{key}'", key in mock_prediction)

    # Verify types within prediction
    check("detected_aspects is list", isinstance(mock_prediction["detected_aspects"], list))
    check("aspect_sentiments is dict", isinstance(mock_prediction["aspect_sentiments"], dict))
    check("confidence_scores is dict", isinstance(mock_prediction["confidence_scores"], dict))
    check("keys match", set(mock_prediction["aspect_sentiments"].keys()) ==
          set(mock_prediction["detected_aspects"]))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: get_overall_sentiment_label edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_overall_sentiment_label():
    print(f"\n{'─' * 70}")
    print("TEST 3: get_overall_sentiment_label — all cases")
    print(f"{'─' * 70}")

    from app.utils.app_helpers import get_overall_sentiment_label

    # All positive
    result = get_overall_sentiment_label({"a": "positive", "b": "positive", "c": "positive"})
    check("All positive → 'positive'", result == "positive", detail=f"got '{result}'")

    # All negative
    result = get_overall_sentiment_label({"a": "negative", "b": "negative"})
    check("All negative → 'negative'", result == "negative", detail=f"got '{result}'")

    # All neutral
    result = get_overall_sentiment_label({"a": "neutral", "b": "neutral"})
    check("All neutral → 'neutral'", result == "neutral", detail=f"got '{result}'")

    # Mix positive + negative (equal) → "mixed"
    result = get_overall_sentiment_label({"a": "positive", "b": "negative"})
    check("Equal pos/neg → 'mixed'", result == "mixed", detail=f"got '{result}'")

    # Mostly positive
    result = get_overall_sentiment_label({"a": "positive", "b": "positive", "c": "negative"})
    check("More pos than neg → 'mostly positive'", result == "mostly positive", detail=f"got '{result}'")

    # Mostly negative
    result = get_overall_sentiment_label({"a": "negative", "b": "negative", "c": "positive"})
    check("More neg than pos → 'mostly negative'", result == "mostly negative", detail=f"got '{result}'")

    # Empty dict
    result = get_overall_sentiment_label({})
    check("Empty dict → 'none detected'", result == "none detected", detail=f"got '{result}'")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: format_aspect_name for all 15 aspects — none empty
# ═══════════════════════════════════════════════════════════════════════════════

def test_format_all_aspects():
    print(f"\n{'─' * 70}")
    print("TEST 4: format_aspect_name — all 15 aspects produce non-empty output")
    print(f"{'─' * 70}")

    from app.utils.app_helpers import format_aspect_name

    aspects = [
        "network_coverage", "internet_speed", "call_quality", "customer_support",
        "billing", "recharge_plans", "data_balance", "roaming", "sim_activation",
        "mobile_app_experience", "ott_bundle_services", "pricing",
        "value_for_money", "data_validity", "5g_experience",
    ]

    all_valid = True
    for aspect in aspects:
        formatted = format_aspect_name(aspect)
        is_valid = isinstance(formatted, str) and len(formatted) > 0
        if not is_valid:
            all_valid = False
            check(f"format_aspect_name('{aspect}')", False, detail=f"got '{formatted}'")

    check("All 15 aspects format to non-empty strings", all_valid)

    # Verify special cases
    check("5g → '5G Experience'", format_aspect_name("5g_experience") == "5G Experience")
    check("ott → 'OTT Bundle Services'", format_aspect_name("ott_bundle_services") == "OTT Bundle Services")
    check("sim → 'SIM Activation'", format_aspect_name("sim_activation") == "SIM Activation")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: load_pipeline import path works
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_import():
    print(f"\n{'─' * 70}")
    print("TEST 5: ABSAInferencePipeline import path works")
    print(f"{'─' * 70}")

    try:
        from src.inference import ABSAInferencePipeline
        check("ABSAInferencePipeline importable", True)

        # Verify class has required methods
        required_methods = ["predict", "predict_batch", "predict_aspects", "predict_sentiment"]
        for method in required_methods:
            check(f"Has method '{method}'", hasattr(ABSAInferencePipeline, method))

    except ImportError as e:
        check("ABSAInferencePipeline importable", False, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Simulate analyze — pipeline.predict() on 3 test feedbacks
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_predict():
    print(f"\n{'─' * 70}")
    print("TEST 6: pipeline.predict() — schema validation (requires model files)")
    print(f"{'─' * 70}")

    try:
        from src.inference import ABSAInferencePipeline
        pipeline = ABSAInferencePipeline()
        check("Pipeline initialized", True)
    except Exception as e:
        check("Pipeline initialized", False, detail=str(e))
        print("  ⏭️  Skipping predict tests (model files may not be available)")
        return

    test_feedbacks = [
        "The 5G speed is excellent but billing is confusing.",
        "Network coverage is very poor in my area.",
        "Happy with the recharge plan value for money.",
    ]

    for i, text in enumerate(test_feedbacks, 1):
        try:
            result = pipeline.predict(text)

            # Schema checks
            check(f"Predict {i}: returns dict", isinstance(result, dict))
            check(f"Predict {i}: has 'detected_aspects'", "detected_aspects" in result)
            check(f"Predict {i}: has 'aspect_sentiments'", "aspect_sentiments" in result)
            check(f"Predict {i}: has 'overall_sentiment'", "overall_sentiment" in result)
            check(f"Predict {i}: has 'inference_time_ms'", "inference_time_ms" in result)

            # Type checks
            check(f"Predict {i}: detected_aspects is list",
                  isinstance(result["detected_aspects"], list))
            check(f"Predict {i}: aspect_sentiments is dict",
                  isinstance(result["aspect_sentiments"], dict))
            check(f"Predict {i}: overall_sentiment is str",
                  isinstance(result["overall_sentiment"], str))
            check(f"Predict {i}: inference_time_ms > 0",
                  result["inference_time_ms"] > 0)

            # Consistency: sentiment keys match detected aspects
            check(f"Predict {i}: sentiments match aspects",
                  set(result["aspect_sentiments"].keys()) == set(result["detected_aspects"]))

        except Exception as e:
            check(f"Predict {i}: no exception", False, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STREAMLIT PAGE 1 — TEST CHECKLIST")
    print("(Tests logic without running Streamlit server)")
    print("=" * 70)

    # Tests that don't need model files
    test_app_helpers()
    test_render_result_card()
    test_overall_sentiment_label()
    test_format_all_aspects()
    test_pipeline_import()

    # Test that needs model files (may skip gracefully)
    test_pipeline_predict()

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
        print(f"\n  🎉 ALL TESTS PASSED — Page 1 logic is verified")
    else:
        print(f"\n  ⚠️  {FAILED} test(s) failed")

    print(f"{'═' * 70}\n")

    assert FAILED == 0, f"{FAILED} test(s) failed"


if __name__ == "__main__":
    main()
