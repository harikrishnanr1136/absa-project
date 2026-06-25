"""
Data Splitting Module for ABSA Telecom Dataset.

Performs stratified train/val/test split preserving multi-label aspect distribution.
Loads paths and ratios from config.yaml.
"""

import json
import logging
import os
import sys
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_stratification_key(aspects_list: list) -> str:
    """
    Create a stratification key from the sorted tuple of aspects.

    For multi-label stratification, we use the sorted aspect combination as a group key.
    This ensures rows with the same aspect combination stay proportionally distributed.

    Args:
        aspects_list: List of aspect strings for a single row

    Returns:
        String representation of sorted aspect tuple (hashable for stratification)
    """
    return str(tuple(sorted(aspects_list)))


def load_and_parse(config: dict) -> pd.DataFrame:
    """Load cleaned CSV and parse JSON columns."""
    data_path = resolve_path(config["data"]["cleaned"])
    logger.info(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def stratified_split(df: pd.DataFrame, config: dict):
    """
    Perform stratified split: 70% train, 15% val, 15% test.

    Uses sorted aspect tuple as stratification key to preserve multi-label distribution.
    Falls back to approximate stratification if too many unique keys cause issues.

    Args:
        df: DataFrame with parsed aspects column
        config: Configuration dict with split ratios and seed

    Returns:
        (train_df, val_df, test_df)
    """
    seed = config["seed"]
    train_ratio = config["split"]["train"]
    val_ratio = config["split"]["val"]
    test_ratio = config["split"]["test"]

    logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {seed}")

    # Create stratification key
    df["_strat_key"] = df["aspects"].apply(create_stratification_key)

    # Handle rare stratification keys (appear only once — can't be split)
    # Group rare keys (count < 3) under a generic label to allow stratification
    key_counts = df["_strat_key"].value_counts()
    rare_keys = key_counts[key_counts < 3].index
    df["_strat_key_safe"] = df["_strat_key"].apply(
        lambda x: "__rare__" if x in rare_keys else x
    )

    logger.info(f"Stratification keys: {df['_strat_key'].nunique()} unique "
                f"({len(rare_keys)} rare keys grouped)")

    try:
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=df["_strat_key_safe"],
        )

        # Second split: val vs test
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=seed,
            stratify=temp_df["_strat_key_safe"],
        )

    except ValueError as e:
        # Fallback: if stratification still fails, use non-stratified split
        logger.warning(f"Stratified split failed ({e}), falling back to random split")
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio), random_state=seed
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test, random_state=seed
        )

    # Drop helper columns
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["_strat_key", "_strat_key_safe"], inplace=True, errors="ignore")

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):
    """Save train/val/test DataFrames to CSV in data/ folder."""
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))

    # Convert lists/dicts back to JSON strings for CSV storage
    for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        split_df = split_df.copy()
        split_df["aspects"] = split_df["aspects"].apply(json.dumps)
        split_df["aspect_sentiments"] = split_df["aspect_sentiments"].apply(json.dumps)

        path = os.path.join(data_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        logger.info(f"Saved {name}.csv: {len(split_df)} rows → {path}")


def print_split_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print detailed split statistics."""
    logger.info(f"\n{'═' * 70}")
    logger.info("DATA SPLIT STATISTICS")
    logger.info(f"{'═' * 70}")

    # Row counts
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"\n{'─' * 70}")
    logger.info("1. ROW COUNTS")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'Split':<10} {'Count':>6} {'Percentage':>12}")
    logger.info(f"  {'─' * 30}")
    logger.info(f"  {'Train':<10} {len(train_df):>6} {len(train_df)*100/total:>10.1f}%")
    logger.info(f"  {'Val':<10} {len(val_df):>6} {len(val_df)*100/total:>10.1f}%")
    logger.info(f"  {'Test':<10} {len(test_df):>6} {len(test_df)*100/total:>10.1f}%")
    logger.info(f"  {'─' * 30}")
    logger.info(f"  {'Total':<10} {total:>6} {100.0:>10.1f}%")

    # Aspect distribution per split
    logger.info(f"\n{'─' * 70}")
    logger.info("2. ASPECT DISTRIBUTION PER SPLIT (%)")
    logger.info(f"{'─' * 70}")

    def get_aspect_dist(df):
        counter = Counter()
        for aspects in df["aspects"]:
            for a in aspects:
                counter[a] += 1
        total_aspects = sum(counter.values())
        return {a: round(c * 100 / total_aspects, 1) for a, c in counter.items()}

    train_dist = get_aspect_dist(train_df)
    val_dist = get_aspect_dist(val_df)
    test_dist = get_aspect_dist(test_df)

    all_aspects = sorted(set(list(train_dist.keys()) + list(val_dist.keys()) + list(test_dist.keys())))

    logger.info(f"\n  {'Aspect':<28} {'Train%':>7} {'Val%':>7} {'Test%':>7} {'Diff':>6}")
    logger.info(f"  {'─' * 57}")
    for aspect in all_aspects:
        t = train_dist.get(aspect, 0)
        v = val_dist.get(aspect, 0)
        te = test_dist.get(aspect, 0)
        max_diff = max(t, v, te) - min(t, v, te)
        flag = " ⚠️" if max_diff > 5 else ""
        logger.info(f"  {aspect:<28} {t:>7.1f} {v:>7.1f} {te:>7.1f} {max_diff:>5.1f}{flag}")

    # Sentiment distribution per split
    logger.info(f"\n{'─' * 70}")
    logger.info("3. SENTIMENT DISTRIBUTION PER SPLIT (%)")
    logger.info(f"{'─' * 70}")

    def get_sentiment_dist(df):
        counter = Counter()
        for sent_dict in df["aspect_sentiments"]:
            for s in sent_dict.values():
                counter[s] += 1
        total_s = sum(counter.values())
        return {s: round(c * 100 / total_s, 1) for s, c in counter.items()}

    train_s = get_sentiment_dist(train_df)
    val_s = get_sentiment_dist(val_df)
    test_s = get_sentiment_dist(test_df)

    logger.info(f"\n  {'Sentiment':<15} {'Train%':>7} {'Val%':>7} {'Test%':>7}")
    logger.info(f"  {'─' * 38}")
    for sent in ["positive", "negative", "neutral"]:
        logger.info(f"  {sent:<15} {train_s.get(sent, 0):>7.1f} {val_s.get(sent, 0):>7.1f} {test_s.get(sent, 0):>7.1f}")

    logger.info(f"\n{'═' * 70}")


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Assert no feedback text appears in more than one split."""
    logger.info(f"\n{'─' * 70}")
    logger.info("4. DATA LEAKAGE CHECK")
    logger.info(f"{'─' * 70}")

    train_texts = set(train_df["feedback"].tolist())
    val_texts = set(val_df["feedback"].tolist())
    test_texts = set(test_df["feedback"].tolist())

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    logger.info(f"  Train ∩ Val overlap:  {len(train_val_overlap)} texts")
    logger.info(f"  Train ∩ Test overlap: {len(train_test_overlap)} texts")
    logger.info(f"  Val ∩ Test overlap:   {len(val_test_overlap)} texts")

    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    if total_overlap == 0:
        logger.info(f"  ✅ NO DATA LEAKAGE — all splits are disjoint")
    else:
        logger.info(f"  ❌ DATA LEAKAGE DETECTED — {total_overlap} overlapping texts")

    assert total_overlap == 0, f"Data leakage: {total_overlap} texts appear in multiple splits"
    logger.info(f"{'═' * 70}\n")


def main():
    logger.info("=" * 70)
    logger.info("DATA SPLIT — Telecom ABSA Dataset")
    logger.info("=" * 70)

    # Load config
    config = load_config()

    # Step 1: Load and parse data
    df = load_and_parse(config)

    # Step 2-3: Stratified split
    train_df, val_df, test_df = stratified_split(df, config)

    # Step 4: Save splits
    save_splits(train_df, val_df, test_df, config)

    # Step 5: Print statistics
    print_split_statistics(train_df, val_df, test_df)

    # Step 6: Verify no leakage
    verify_no_leakage(train_df, val_df, test_df)

    logger.info("Data split complete.")


if __name__ == "__main__":
    main()
