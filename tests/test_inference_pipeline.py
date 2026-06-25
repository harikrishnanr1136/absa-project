"""
Production Readiness Tests for ABSAInferencePipeline.

Verifies that src/inference.py is fully functional before Streamlit deployment.
Covers: initialization, standard inputs, edge cases, batch processing.
"""

import logging
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Track results
PASSED = 0
FAILED = 0
FAILURES = []

VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed", "none"}


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


def validate_prediction(result: dict, input_text: str, test_label: str):
    """Validate a single prediction result against expected schema."""
    # detected_aspects is a list
    check(f"{test_label}: detected_aspects is list",
          isinstance(result.get("detected_aspects"), list),
          detail=f"Got type: {type(result.get('detected_aspects'))}")

    detected = result.get("detected_aspects", [])
    sentiments = result.get("aspect_sentiments", {})
    confidences = result.get("confidence_scores", {})

    # aspect_sentiments keys match detected_aspects
    check(f"{test_label}: aspect_sentiments keys match detected_aspects",
          set(sentiments.keys()) == set(detected),
          detail=f"sentiments={set(sentiments.keys())}, detected={set(detected)}")

    # confidence_scores keys match detected_aspects
    check(f"{test_label}: confidence_scores keys match detected_aspects",
          set(confidences.keys()) == set(detected),
          detail=f"confidences={set(confidences.keys())}, detected={set(detected)}")

    # overall_sentiment is valid
    overall = result.get("overall_sentiment", "")
    check(f"{test_label}: overall_sentiment is valid",
          overall in VALID_SENTIMENTS,
          detail=f"Got: '{overall}'")

    # inference_time_ms is positive float
    inf_time = result.get("inference_time_ms", -1)
    check(f"{test_label}: inference_time_ms is positive",
          isinstance(inf_time, (int, float)) and inf_time > 0,
          detail=f"Got: {inf_time}")


def main():
    global PASSED, FAILED

    print("=" * 70)
    print("INFERENCE PIPELINE — PRODUCTION READINESS TESTS")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Import without errors
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TEST 1: Import ABSAInferencePipeline")
    print(f"{'─' * 70}")

    try:
        from src.inference import ABSAInferencePipeline
        check("Import successful", True)
    except Exception as e:
        check("Import successful", False, detail=str(e))
        print("\n⛔ Cannot proceed — import failed")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Initialize pipeline
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TEST 2: Initialize Pipeline (measure load time)")
    print(f"{'─' * 70}")

    try:
        start = time.time()
        pipeline = ABSAInferencePipeline()
        load_time = time.time() - start
        check("Pipeline initialized", True)
        print(f"  ⏱  Load time: {load_time:.2f}s")
    except Exception as e:
        check("Pipeline initialized", False, detail=str(e))
        print("\n⛔ Cannot proceed — initialization failed")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Standard predictions (10 inputs)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TEST 3: Standard Predictions (10 inputs)")
    print(f"{'─' * 70}")

    standard_inputs = [
        "The internet speed is excellent but billing is very confusing.",
        "Network coverage is poor in my area.",
        "Customer support was helpful and resolved my issue quickly.",
        "5G experience is amazing, worth the upgrade.",
        "Recharge plans are too expensive compared to competitors.",
        "App keeps crashing during data balance check.",
        "SIM activation took 3 days, totally unacceptable.",
        "OTT bundle is great value for money.",
        "Roaming charges r way too high omg",
        "ok i guess",
    ]

    standard_results = []
    for i, text in enumerate(standard_inputs, 1):
        try:
            result = pipeline.predict(text)
            standard_results.append(result)
            validate_prediction(result, text, f"Input {i}")
            print(f"      → Aspects: {result['detected_aspects']}, "
                  f"Overall: {result['overall_sentiment']}, "
                  f"Time: {result['inference_time_ms']:.1f}ms")
        except Exception as e:
            check(f"Input {i}: no exception", False, detail=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Edge cases
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TEST 4: Edge Cases")
    print(f"{'─' * 70}")

    # Generate very long feedback (200+ words)
    long_feedback = (
        "I have been using this telecom service for over two years now and I must say "
        "the experience has been extremely inconsistent across all dimensions. The network "
        "coverage in my residential area is decent during daytime hours but completely falls "
        "apart after 8 PM when everyone starts streaming. The internet speed drops to "
        "practically unusable levels and I cannot even load basic web pages let alone stream "
        "any video content. The customer support team is polite but never actually resolves "
        "anything. Every time I call they just log a ticket and promise a callback that never "
        "comes. The billing system has overcharged me twice in the last six months and getting "
        "refunds processed takes weeks of follow-up. The only positive thing I can say is that "
        "the recharge plans offer decent data allocations and the OTT bundle with Netflix "
        "included is genuinely good value. The mobile app works most of the time but crashes "
        "occasionally when checking data balance. Overall I feel stuck because switching "
        "carriers means losing my number and dealing with SIM activation hassles again."
    )

    edge_cases = [
        ("Empty string", ""),
        ("Single word", "bad"),
        ("Very long (200+ words)", long_feedback),
        ("Repeated text", "good good good good good"),
        ("All uppercase", "NETWORK IS VERY BAD"),
    ]

    edge_results = []
    for label, text in edge_cases:
        try:
            result = pipeline.predict(text)
            edge_results.append(result)
            validate_prediction(result, text, f"Edge: {label}")
            print(f"      → Aspects: {result['detected_aspects']}, "
                  f"Overall: {result['overall_sentiment']}")
        except Exception as e:
            check(f"Edge: {label}: no exception", False, detail=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: Batch prediction
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("TEST 5: Batch Prediction (all 15 inputs combined)")
    print(f"{'─' * 70}")

    all_inputs = standard_inputs + [text for _, text in edge_cases]

    try:
        start = time.time()
        batch_results = pipeline.predict_batch(all_inputs)
        batch_time = (time.time() - start) * 1000

        check("predict_batch returns list", isinstance(batch_results, list))
        check("Output length matches input",
              len(batch_results) == len(all_inputs),
              detail=f"in={len(all_inputs)}, out={len(batch_results)}")
        check("No None results", all(r is not None for r in batch_results))

        per_sample = batch_time / len(all_inputs)
        print(f"  ⏱  Total: {batch_time:.1f}ms | Per-sample: {per_sample:.1f}ms")

    except Exception as e:
        check("predict_batch: no exception", False, detail=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("TEST SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  Total tests:  {PASSED + FAILED}")
    print(f"  Passed:       {PASSED}")
    print(f"  Failed:       {FAILED}")

    if FAILURES:
        print(f"\n  FAILURES:")
        for f in FAILURES:
            print(f"    • {f}")

    if FAILED == 0:
        print(f"\n  🎉 ALL TESTS PASSED — Pipeline is production-ready")
    else:
        print(f"\n  ⚠️  {FAILED} test(s) failed — fix before deploying to Streamlit")

    print(f"{'═' * 70}\n")

    assert FAILED == 0, f"{FAILED} test(s) failed"


if __name__ == "__main__":
    main()
