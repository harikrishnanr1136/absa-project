"""
CSV Helper Utilities for Batch Processing.

Provides validation, cleaning, result formatting, and summary statistics
for the batch CSV processing page in the Streamlit ABSA app.
"""

import json
import logging
from collections import Counter
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Maximum rows allowed for batch processing
MAX_ROWS = 1000

# Accepted column names for feedback (case-insensitive matching)
ACCEPTED_FEEDBACK_COLUMNS = ["feedback", "text", "review", "comment", "message"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. validate_csv
# ═══════════════════════════════════════════════════════════════════════════════

def validate_csv(df: pd.DataFrame) -> dict:
    """
    Validate an uploaded CSV dataframe for batch inference.

    Checks:
    - Has a recognizable feedback column (case-insensitive)
    - Not empty
    - At least 1 valid (non-null, non-empty) feedback row
    - Row count does not exceed MAX_ROWS (1000)

    Args:
        df: Raw pandas DataFrame from CSV upload

    Returns:
        Dict with keys: valid, error, warning, row_count, valid_row_count,
        invalid_row_count, column_used
    """
    result = {
        "valid": False,
        "error": None,
        "warning": None,
        "row_count": 0,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "column_used": None,
    }

    # Check dataframe is not empty
    if df is None or df.empty:
        result["error"] = "Uploaded CSV is empty. Please provide a file with at least one row."
        logger.warning("CSV validation failed: empty dataframe")
        return result

    result["row_count"] = len(df)

    # Find feedback column (case-insensitive)
    matched_column = None
    df_columns_lower = {col.lower().strip(): col for col in df.columns}

    for accepted in ACCEPTED_FEEDBACK_COLUMNS:
        if accepted in df_columns_lower:
            matched_column = df_columns_lower[accepted]
            break

    if matched_column is None:
        result["error"] = (
            f"No feedback column found. Expected one of: {ACCEPTED_FEEDBACK_COLUMNS} "
            f"(case-insensitive). Found columns: {list(df.columns)}"
        )
        logger.warning(f"CSV validation failed: no feedback column. Columns: {list(df.columns)}")
        return result

    result["column_used"] = matched_column

    # Rename to "feedback" if needed
    if matched_column != "feedback":
        df.rename(columns={matched_column: "feedback"}, inplace=True)
        logger.info(f"Renamed column '{matched_column}' to 'feedback'")

    # Count valid rows (non-null, non-empty after strip)
    valid_mask = df["feedback"].notna() & (df["feedback"].astype(str).str.strip() != "")
    valid_count = valid_mask.sum()
    invalid_count = len(df) - valid_count

    result["valid_row_count"] = int(valid_count)
    result["invalid_row_count"] = int(invalid_count)

    if valid_count == 0:
        result["error"] = "No valid feedback rows found. All rows are empty or null."
        logger.warning("CSV validation failed: no valid rows")
        return result

    # Check row limit
    if len(df) > MAX_ROWS:
        result["error"] = (
            f"CSV has {len(df)} rows, exceeding the maximum limit of {MAX_ROWS}. "
            f"Please reduce the file size."
        )
        logger.warning(f"CSV validation failed: {len(df)} rows exceeds limit {MAX_ROWS}")
        return result

    # Warnings
    if invalid_count > 0:
        result["warning"] = (
            f"{invalid_count} row(s) have empty/null feedback and will be skipped."
        )

    result["valid"] = True
    logger.info(f"CSV validation passed: {valid_count} valid rows, {invalid_count} invalid")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. clean_csv_for_inference
# ═══════════════════════════════════════════════════════════════════════════════

def clean_csv_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean uploaded dataframe for inference.

    Steps:
    - Keep only the feedback column
    - Drop rows where feedback is null or empty after strip
    - Reset index
    - Add original_row_number column (1-based)

    Args:
        df: DataFrame with 'feedback' column (after validation/rename)

    Returns:
        Cleaned DataFrame with columns: original_row_number, feedback
    """
    # Keep only feedback
    cleaned = df[["feedback"]].copy()

    # Add original row number (1-based, before dropping)
    cleaned["original_row_number"] = range(1, len(cleaned) + 1)

    # Drop invalid rows
    cleaned["feedback"] = cleaned["feedback"].astype(str).str.strip()
    cleaned = cleaned[cleaned["feedback"] != ""]
    cleaned = cleaned[cleaned["feedback"] != "nan"]
    cleaned = cleaned[cleaned["feedback"].notna()]

    # Reset index
    cleaned = cleaned.reset_index(drop=True)

    logger.info(f"Cleaned CSV: {len(cleaned)} rows ready for inference")
    return cleaned[["original_row_number", "feedback"]]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. prepare_results_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_results_dataframe(df: pd.DataFrame, predictions: List[dict]) -> pd.DataFrame:
    """
    Combine original cleaned dataframe with prediction results.

    Args:
        df: Cleaned DataFrame with original_row_number and feedback
        predictions: List of prediction dicts from pipeline.predict_batch()

    Returns:
        Results DataFrame with columns: original_row_number, feedback,
        detected_aspects, aspect_sentiments, confidence_scores,
        overall_sentiment, inference_time_ms, aspect_count
    """
    rows = []
    for i, pred in enumerate(predictions):
        detected = pred.get("detected_aspects", [])
        rows.append({
            "original_row_number": df.iloc[i]["original_row_number"] if i < len(df) else i + 1,
            "feedback": pred.get("feedback", ""),
            "detected_aspects": ", ".join(detected),
            "aspect_sentiments": json.dumps(pred.get("aspect_sentiments", {})),
            "confidence_scores": json.dumps(pred.get("confidence_scores", {})),
            "overall_sentiment": pred.get("overall_sentiment", "neutral"),
            "inference_time_ms": round(pred.get("inference_time_ms", 0.0), 2),
            "aspect_count": len(detected),
        })

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("original_row_number").reset_index(drop=True)

    logger.info(f"Results dataframe prepared: {len(results_df)} rows")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. results_to_download_csv
# ═══════════════════════════════════════════════════════════════════════════════

def results_to_download_csv(results_df: pd.DataFrame) -> bytes:
    """
    Convert results dataframe to CSV bytes for st.download_button.

    Args:
        results_df: DataFrame from prepare_results_dataframe()

    Returns:
        UTF-8 encoded CSV bytes
    """
    csv_str = results_df.to_csv(index=False, encoding="utf-8")
    return csv_str.encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. get_batch_summary_stats
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch_summary_stats(results_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for batch processing results.

    Args:
        results_df: DataFrame from prepare_results_dataframe()

    Returns:
        Dict with: total_rows, rows_with_aspects, rows_without_aspects,
        sentiment_distribution, most_common_aspect, most_common_positive_aspect,
        most_common_negative_aspect, avg_aspects_per_feedback, avg_inference_time_ms
    """
    total = len(results_df)

    # Rows with/without aspects
    rows_with = int((results_df["aspect_count"] > 0).sum())
    rows_without = total - rows_with

    # Sentiment distribution
    sent_counts = results_df["overall_sentiment"].value_counts().to_dict()
    sent_pcts = {k: round(v * 100 / max(total, 1), 1) for k, v in sent_counts.items()}

    # Aspect frequencies and sentiment-specific aspects
    all_aspects = Counter()
    positive_aspects = Counter()
    negative_aspects = Counter()

    for _, row in results_df.iterrows():
        aspects_str = row["detected_aspects"]
        if not aspects_str or aspects_str == "":
            continue

        aspects = [a.strip() for a in aspects_str.split(",") if a.strip()]
        all_aspects.update(aspects)

        # Parse sentiments
        try:
            sentiments = json.loads(row["aspect_sentiments"])
            for aspect, sentiment in sentiments.items():
                if sentiment == "positive":
                    positive_aspects[aspect] += 1
                elif sentiment == "negative":
                    negative_aspects[aspect] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Most common aspects
    most_common = all_aspects.most_common(1)[0][0] if all_aspects else None
    most_positive = positive_aspects.most_common(1)[0][0] if positive_aspects else None
    most_negative = negative_aspects.most_common(1)[0][0] if negative_aspects else None

    # Averages
    avg_aspects = round(results_df["aspect_count"].mean(), 2) if total > 0 else 0.0
    avg_time = round(results_df["inference_time_ms"].mean(), 2) if total > 0 else 0.0

    stats = {
        "total_rows": total,
        "rows_with_aspects": rows_with,
        "rows_without_aspects": rows_without,
        "sentiment_distribution": {
            "counts": sent_counts,
            "percentages": sent_pcts,
        },
        "most_common_aspect": most_common,
        "most_common_positive_aspect": most_positive,
        "most_common_negative_aspect": most_negative,
        "avg_aspects_per_feedback": avg_aspects,
        "avg_inference_time_ms": avg_time,
    }

    logger.info(f"Batch stats: {total} rows, {rows_with} with aspects, "
                f"avg {avg_aspects} aspects/row, avg {avg_time}ms/row")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Test with mock data
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("csv_helpers.py — FUNCTION TESTS")
    print("=" * 60)

    # ─── Mock Data ────────────────────────────────────────────────────────
    mock_df = pd.DataFrame({
        "Feedback": [
            "Network coverage is excellent in my area.",
            "Terrible internet speed since last week.",
            "",
            "Customer support was very helpful today.",
            None,
            "5G is fast but billing is confusing.",
        ]
    })

    mock_predictions = [
        {
            "feedback": "Network coverage is excellent in my area.",
            "detected_aspects": ["network_coverage"],
            "aspect_sentiments": {"network_coverage": "positive"},
            "confidence_scores": {"network_coverage": 0.92},
            "overall_sentiment": "positive",
            "inference_time_ms": 120.5,
        },
        {
            "feedback": "Terrible internet speed since last week.",
            "detected_aspects": ["internet_speed"],
            "aspect_sentiments": {"internet_speed": "negative"},
            "confidence_scores": {"internet_speed": 0.88},
            "overall_sentiment": "negative",
            "inference_time_ms": 115.2,
        },
        {
            "feedback": "Customer support was very helpful today.",
            "detected_aspects": ["customer_support"],
            "aspect_sentiments": {"customer_support": "positive"},
            "confidence_scores": {"customer_support": 0.85},
            "overall_sentiment": "positive",
            "inference_time_ms": 98.7,
        },
        {
            "feedback": "5G is fast but billing is confusing.",
            "detected_aspects": ["5g_experience", "billing"],
            "aspect_sentiments": {"5g_experience": "positive", "billing": "negative"},
            "confidence_scores": {"5g_experience": 0.90, "billing": 0.76},
            "overall_sentiment": "mixed",
            "inference_time_ms": 145.3,
        },
    ]

    # ─── Test 1: validate_csv ─────────────────────────────────────────────
    print("\n1. validate_csv:")
    result = validate_csv(mock_df)
    print(f"   valid: {result['valid']}")
    print(f"   column_used: {result['column_used']}")
    print(f"   row_count: {result['row_count']}")
    print(f"   valid_row_count: {result['valid_row_count']}")
    print(f"   invalid_row_count: {result['invalid_row_count']}")
    print(f"   warning: {result['warning']}")
    assert result["valid"] is True
    assert result["column_used"] == "Feedback"
    assert result["valid_row_count"] == 4
    print("   ✅ PASSED")

    # Test with missing column
    bad_df = pd.DataFrame({"name": ["Alice"]})
    result = validate_csv(bad_df)
    assert result["valid"] is False
    print("   ✅ Missing column detected correctly")

    # Test with empty df
    result = validate_csv(pd.DataFrame())
    assert result["valid"] is False
    print("   ✅ Empty dataframe detected correctly")

    # ─── Test 2: clean_csv_for_inference ──────────────────────────────────
    print("\n2. clean_csv_for_inference:")
    # Re-create mock with renamed column
    mock_df2 = pd.DataFrame({"feedback": [
        "Network coverage is excellent.", "Terrible speed.", "", "Support was helpful.", None, "5G fast."
    ]})
    cleaned = clean_csv_for_inference(mock_df2)
    print(f"   Input rows: 6, Output rows: {len(cleaned)}")
    print(f"   Columns: {list(cleaned.columns)}")
    assert len(cleaned) == 4
    assert "original_row_number" in cleaned.columns
    assert "feedback" in cleaned.columns
    print("   ✅ PASSED")

    # ─── Test 3: prepare_results_dataframe ────────────────────────────────
    print("\n3. prepare_results_dataframe:")
    cleaned_for_results = pd.DataFrame({
        "original_row_number": [1, 2, 4, 6],
        "feedback": ["Net coverage.", "Speed.", "Support.", "5G."],
    })
    results_df = prepare_results_dataframe(cleaned_for_results, mock_predictions)
    print(f"   Columns: {list(results_df.columns)}")
    print(f"   Rows: {len(results_df)}")
    assert "aspect_count" in results_df.columns
    assert "overall_sentiment" in results_df.columns
    assert len(results_df) == 4
    print("   ✅ PASSED")

    # ─── Test 4: results_to_download_csv ──────────────────────────────────
    print("\n4. results_to_download_csv:")
    csv_bytes = results_to_download_csv(results_df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0
    assert b"overall_sentiment" in csv_bytes
    print(f"   Size: {len(csv_bytes)} bytes")
    print("   ✅ PASSED")

    # ─── Test 5: get_batch_summary_stats ──────────────────────────────────
    print("\n5. get_batch_summary_stats:")
    stats = get_batch_summary_stats(results_df)
    print(f"   total_rows: {stats['total_rows']}")
    print(f"   rows_with_aspects: {stats['rows_with_aspects']}")
    print(f"   most_common_aspect: {stats['most_common_aspect']}")
    print(f"   most_common_positive: {stats['most_common_positive_aspect']}")
    print(f"   most_common_negative: {stats['most_common_negative_aspect']}")
    print(f"   avg_aspects: {stats['avg_aspects_per_feedback']}")
    print(f"   avg_time: {stats['avg_inference_time_ms']}ms")
    print(f"   sentiment_dist: {stats['sentiment_distribution']['counts']}")
    assert stats["total_rows"] == 4
    assert stats["rows_with_aspects"] == 4
    print("   ✅ PASSED")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print("=" * 60)
