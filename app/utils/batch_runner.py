"""
Batch Inference Runner with Progress Tracking for Streamlit.

Runs inference row-by-row with progress bar updates and graceful error handling.
Never lets one row failure stop the batch.
"""

import logging
import time
from typing import List

logger = logging.getLogger(__name__)


def run_batch_inference(pipeline, feedbacks: List[str], progress_bar, status_text) -> List[dict]:
    """
    Run inference on a list of feedbacks with Streamlit progress tracking.

    Iterates over each feedback, calls pipeline.predict(), and updates the
    progress bar and status text. If a single row fails, logs the error and
    appends an error record — never stops the batch.

    Args:
        pipeline: Loaded ABSAInferencePipeline instance
        feedbacks: List of feedback strings to analyze
        progress_bar: Streamlit st.progress() element
        status_text: Streamlit st.empty() or st.text() element for status updates

    Returns:
        List of prediction dicts (same length as feedbacks).
        Failed rows have overall_sentiment="error" and an "error" key.
    """
    total = len(feedbacks)
    results = []
    error_count = 0

    logger.info(f"Batch inference starting: {total} feedbacks")
    start_time = time.time()

    for i, feedback in enumerate(feedbacks):
        # Update progress
        progress_bar.progress((i) / total)
        preview = feedback[:50] + "..." if len(feedback) > 50 else feedback
        status_text.text(f"Analyzing {i + 1}/{total} — {preview}")

        try:
            prediction = pipeline.predict(feedback)
            results.append(prediction)

        except Exception as e:
            # Log error but continue batch
            error_count += 1
            logger.error(f"Row {i + 1} failed: {str(e)} | Feedback: '{feedback[:80]}'")

            # Append error record with consistent schema
            results.append({
                "feedback": feedback,
                "detected_aspects": [],
                "aspect_sentiments": {},
                "confidence_scores": {},
                "overall_sentiment": "error",
                "inference_time_ms": 0,
                "error": str(e),
            })

    # Finalize progress
    progress_bar.progress(1.0)

    elapsed = time.time() - start_time
    status_text.text(f"Complete — {total} feedbacks analyzed in {elapsed:.1f}s.")

    if error_count > 0:
        logger.warning(f"Batch completed with {error_count} error(s) out of {total} rows")
    else:
        logger.info(f"Batch completed successfully: {total} rows in {elapsed:.1f}s")

    return results


def estimate_batch_time(n_rows: int, avg_inference_ms: float = 500.0) -> str:
    """
    Estimate total time for a batch of n_rows.

    Args:
        n_rows: Number of rows to process
        avg_inference_ms: Average inference time per sample in milliseconds.
                         Defaults to 500ms (conservative estimate for DistilBERT on CPU).

    Returns:
        Human-readable time estimate string (e.g., "~45 seconds" or "~2 minutes")
    """
    total_ms = n_rows * avg_inference_ms
    total_seconds = total_ms / 1000

    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        if minutes < 2:
            return "~1 minute"
        else:
            return f"~{int(minutes)} minutes"
    else:
        hours = total_seconds / 3600
        return f"~{hours:.1f} hours"


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Test without Streamlit
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("batch_runner.py — FUNCTION TESTS")
    print("=" * 60)

    # Test estimate_batch_time
    print("\n1. estimate_batch_time:")
    test_cases = [
        (10, 500),
        (50, 500),
        (100, 500),
        (200, 500),
        (500, 500),
        (1000, 500),
        (20, 100),
    ]
    for n, ms in test_cases:
        print(f"   {n} rows @ {ms}ms/row → {estimate_batch_time(n, ms)}")

    # Test run_batch_inference with mock objects
    print("\n2. run_batch_inference (mock pipeline + mock progress):")

    # Mock pipeline
    class MockPipeline:
        def predict(self, text):
            if "FAIL" in text:
                raise ValueError("Simulated failure")
            return {
                "feedback": text,
                "detected_aspects": ["network_coverage"],
                "aspect_sentiments": {"network_coverage": "positive"},
                "confidence_scores": {"network_coverage": 0.9},
                "overall_sentiment": "positive",
                "inference_time_ms": 50.0,
            }

    # Mock Streamlit progress/status
    class MockProgress:
        def progress(self, value):
            pass

    class MockStatus:
        def text(self, msg):
            pass

    feedbacks = [
        "Great network coverage.",
        "Internet speed is terrible.",
        "FAIL this should error gracefully.",
        "Customer support was helpful.",
    ]

    results = run_batch_inference(MockPipeline(), feedbacks, MockProgress(), MockStatus())

    print(f"   Input: {len(feedbacks)} rows")
    print(f"   Output: {len(results)} results")
    print(f"   Errors: {sum(1 for r in results if r['overall_sentiment'] == 'error')}")

    assert len(results) == len(feedbacks), "Output length mismatch"
    assert results[2]["overall_sentiment"] == "error", "Error row not marked"
    assert "error" in results[2], "Error record missing 'error' key"
    assert results[0]["overall_sentiment"] == "positive", "Success row incorrect"
    print("   ✅ All assertions passed")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print("=" * 60)
