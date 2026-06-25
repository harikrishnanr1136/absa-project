"""
Handle Missing Values in Telecom ABSA Dataset
Loads CSV, identifies and drops rows with missing/empty values, saves cleaned output.
"""

import json
import logging
import os
import sys

import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "absa_telecom_combined.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "telecom_absa_cleaned.csv")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset with error handling."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset from: {path}")
        logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)


def report_missing_values(df: pd.DataFrame):
    """Check and report missing values per column."""
    logger.info("─" * 60)
    logger.info("MISSING VALUE REPORT")
    logger.info("─" * 60)

    for col in df.columns:
        null_count = df[col].isna().sum()
        if col == "feedback":
            empty_count = df[col].fillna("").apply(lambda x: x.strip() == "").sum()
        elif col in ("aspects", "aspect_sentiments"):
            empty_count = df[col].fillna("").apply(
                lambda x: _is_empty_json(x)
            ).sum()
        else:
            empty_count = 0

        total = null_count + empty_count - min(null_count, empty_count)  # avoid double count
        status = "CLEAN" if null_count == 0 and empty_count == 0 else "HAS ISSUES"
        logger.info(f"  {col:<22} null={null_count:<4} empty={empty_count:<4} [{status}]")


def _is_empty_json(val) -> bool:
    """Check if a JSON string value parses to an empty list or dict."""
    if pd.isna(val) or val == "":
        return True
    try:
        parsed = json.loads(val)
        if isinstance(parsed, (list, dict)) and len(parsed) == 0:
            return True
    except (json.JSONDecodeError, TypeError):
        return True
    return False


def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where feedback is None, NaN, or empty string after stripping."""
    before = len(df)

    # Identify rows to drop
    mask = df["feedback"].isna() | df["feedback"].apply(
        lambda x: str(x).strip() == "" if pd.notna(x) else True
    )
    drop_count = mask.sum()

    df = df[~mask].copy()
    logger.info(f"  Feedback: dropped {drop_count} rows (empty/null feedback)")
    logger.info(f"    Rows: {before} -> {len(df)}")
    return df


def clean_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where aspects is empty list after JSON parsing."""
    before = len(df)

    mask = df["aspects"].apply(_is_empty_json)
    drop_count = mask.sum()

    df = df[~mask].copy()
    logger.info(f"  Aspects: dropped {drop_count} rows (empty aspects list)")
    logger.info(f"    Rows: {before} -> {len(df)}")
    return df


def clean_aspect_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where aspect_sentiments dict is empty after JSON parsing."""
    before = len(df)

    mask = df["aspect_sentiments"].apply(_is_empty_json)
    drop_count = mask.sum()

    df = df[~mask].copy()
    logger.info(f"  Aspect sentiments: dropped {drop_count} rows (empty sentiments dict)")
    logger.info(f"    Rows: {before} -> {len(df)}")
    return df


def save_dataset(df: pd.DataFrame, path: str):
    """Save cleaned dataset to CSV with error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved cleaned dataset to: {path}")
        logger.info(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    except PermissionError:
        logger.error(f"Permission denied writing to: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        sys.exit(1)


def main():
    logger.info("=" * 60)
    logger.info("MISSING VALUE HANDLING - Telecom ABSA Dataset")
    logger.info("=" * 60)

    # Step 1: Load dataset
    df = load_dataset(INPUT_PATH)
    initial_count = len(df)

    # Step 2: Report missing values
    report_missing_values(df)

    # Steps 3-5: Clean each column
    logger.info("")
    logger.info("─" * 60)
    logger.info("CLEANING OPERATIONS")
    logger.info("─" * 60)

    df = clean_feedback(df)
    df = clean_aspects(df)
    df = clean_aspect_sentiments(df)

    # Step 6: Reset index
    df = df.reset_index(drop=True)
    logger.info(f"  Index reset after dropping rows")

    # Step 7: Assert no missing values remain
    logger.info("")
    logger.info("─" * 60)
    logger.info("POST-CLEANING VALIDATION")
    logger.info("─" * 60)

    feedback_missing = df["feedback"].isna().sum() + df["feedback"].apply(
        lambda x: str(x).strip() == ""
    ).sum()
    aspects_missing = df["aspects"].apply(_is_empty_json).sum()
    sentiments_missing = df["aspect_sentiments"].apply(_is_empty_json).sum()

    assert feedback_missing == 0, f"Still have {feedback_missing} missing feedback values"
    assert aspects_missing == 0, f"Still have {aspects_missing} empty aspects"
    assert sentiments_missing == 0, f"Still have {sentiments_missing} empty sentiments"

    logger.info("  ✓ No missing feedback values")
    logger.info("  ✓ No empty aspects lists")
    logger.info("  ✓ No empty aspect_sentiments dicts")
    logger.info("  ✓ All assertions passed")

    # Step 8: Save cleaned dataset
    logger.info("")
    logger.info("─" * 60)
    logger.info("SAVING CLEANED DATASET")
    logger.info("─" * 60)
    save_dataset(df, OUTPUT_PATH)

    # Step 9: Print summary
    final_count = len(df)
    dropped_total = initial_count - final_count

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Before: {initial_count} rows")
    logger.info(f"  After:  {final_count} rows")
    logger.info(f"  Dropped: {dropped_total} rows ({dropped_total * 100 / initial_count:.1f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
