"""
Data Preparation for DistilBERT Fine-Tuning on Telecom ABSA.

Prepares HuggingFace Dataset objects for:
  Task 1: Aspect Detection (multi-label, 15 classes)
  Task 2: Per-Aspect Sentiment Classification (3 classes per aspect)

Tokenizes with DistilBertTokenizer (max_length=128).
Saves all datasets to data/hf_datasets/.
"""

import json
import logging
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizer

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTIMENT_MAP = {"positive": 0, "negative": 1, "neutral": 2}


def load_split(path: str) -> pd.DataFrame:
    """Load CSV and parse JSON columns."""
    df = pd.read_csv(path)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)
    return df


def build_aspect_detection_dataset(train_df, val_df, test_df, mlb, tokenizer) -> DatasetDict:
    """
    Task 1: Build HuggingFace DatasetDict for multi-label aspect detection.

    Each example: {"text": str, "labels": list[float] (binary vector of 15)}
    """
    logger.info("Building Aspect Detection datasets...")

    def df_to_hf(df, split_name):
        texts = df["feedback"].tolist()
        labels = mlb.transform(df["aspects"]).astype(np.float32).tolist()

        ds = Dataset.from_dict({"text": texts, "labels": labels})
        logger.info(f"  {split_name}: {len(ds)} samples")
        return ds

    ds_dict = DatasetDict({
        "train": df_to_hf(train_df, "train"),
        "val": df_to_hf(val_df, "val"),
        "test": df_to_hf(test_df, "test"),
    })

    # Tokenize
    logger.info("  Tokenizing aspect detection datasets...")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )

    ds_dict = ds_dict.map(tokenize_fn, batched=True, desc="Tokenizing")

    return ds_dict


def build_sentiment_datasets(train_df, val_df, test_df, aspect_labels, tokenizer) -> dict:
    """
    Task 2: For each of the 15 aspects, build a separate DatasetDict for
    sentiment classification (positive=0, negative=1, neutral=2).
    """
    logger.info("Building Per-Aspect Sentiment datasets...")

    all_aspect_datasets = {}

    for aspect in aspect_labels:
        logger.info(f"\n  Aspect: {aspect}")

        splits = {}
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            # Filter rows containing this aspect
            mask = df["aspects"].apply(lambda a: aspect in a)
            subset = df[mask]

            if len(subset) == 0:
                logger.warning(f"    {split_name}: 0 samples — skipping")
                continue

            texts = subset["feedback"].tolist()
            labels = subset["aspect_sentiments"].apply(
                lambda d: SENTIMENT_MAP.get(d.get(aspect, "neutral"), 2)
            ).tolist()

            splits[split_name] = Dataset.from_dict({"text": texts, "label": labels})
            dist = Counter(labels)
            logger.info(f"    {split_name}: {len(texts)} samples — "
                        f"pos={dist.get(0,0)}, neg={dist.get(1,0)}, neu={dist.get(2,0)}")

        if not splits:
            logger.warning(f"  No data for aspect '{aspect}' — skipping entirely")
            continue

        ds_dict = DatasetDict(splits)

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
            )

        ds_dict = ds_dict.map(tokenize_fn, batched=True, desc=f"Tokenizing {aspect}")
        all_aspect_datasets[aspect] = ds_dict

    return all_aspect_datasets


def main():
    logger.info("=" * 70)
    logger.info("DL DATA PREPARATION — DistilBERT Fine-Tuning")
    logger.info("=" * 70)

    # ─── Config ───────────────────────────────────────────────────────────
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    hf_dataset_dir = os.path.join(data_dir, "hf_datasets")
    os.makedirs(hf_dataset_dir, exist_ok=True)

    # ─── Load Splits ──────────────────────────────────────────────────────
    logger.info("Loading data splits...")
    train_df = load_split(os.path.join(data_dir, "train.csv"))
    val_df = load_split(os.path.join(data_dir, "val.csv"))
    test_df = load_split(os.path.join(data_dir, "test.csv"))
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ─── Load/Reuse MultiLabelBinarizer ───────────────────────────────────
    mlb_path = os.path.join(output_dir, "mlb.pkl")
    if os.path.exists(mlb_path):
        mlb = joblib.load(mlb_path)
        logger.info(f"Loaded existing MLB from: {mlb_path}")
    else:
        mlb = MultiLabelBinarizer(classes=aspect_labels)
        mlb.fit([aspect_labels])
        joblib.dump(mlb, mlb_path)
        logger.info(f"Created and saved new MLB to: {mlb_path}")

    logger.info(f"MLB classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

    # ─── Load Tokenizer ───────────────────────────────────────────────────
    logger.info("Loading DistilBertTokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    logger.info(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size}")

    # ─── Task 1: Aspect Detection Dataset ─────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("TASK 1: Aspect Detection (Multi-Label)")
    logger.info("─" * 70)

    aspect_ds = build_aspect_detection_dataset(train_df, val_df, test_df, mlb, tokenizer)

    aspect_ds_path = os.path.join(hf_dataset_dir, "aspect_detection")
    aspect_ds.save_to_disk(aspect_ds_path)
    logger.info(f"Saved: {aspect_ds_path}")

    # ─── Task 2: Sentiment Classification Datasets ────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("TASK 2: Per-Aspect Sentiment Classification")
    logger.info("─" * 70)

    sentiment_datasets = build_sentiment_datasets(
        train_df, val_df, test_df, aspect_labels, tokenizer
    )

    for aspect, ds_dict in sentiment_datasets.items():
        aspect_path = os.path.join(hf_dataset_dir, f"sentiment_{aspect}")
        ds_dict.save_to_disk(aspect_path)

    logger.info(f"\nSaved {len(sentiment_datasets)} aspect sentiment datasets to: {hf_dataset_dir}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("DATA PREPARATION SUMMARY")
    print(f"{'═' * 70}")

    print(f"\n{'─' * 70}")
    print("Task 1: Aspect Detection")
    print(f"{'─' * 70}")
    print(f"  {'Split':<8} {'Samples':>8}")
    print(f"  {'─' * 18}")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name:<8} {len(aspect_ds[split_name]):>8}")

    print(f"\n{'─' * 70}")
    print("Task 2: Sentiment Classification (per aspect)")
    print(f"{'─' * 70}")
    print(f"\n  {'Aspect':<28} {'Train':>6} {'Val':>6} {'Test':>6}")
    print(f"  {'─' * 50}")
    for aspect in aspect_labels:
        if aspect in sentiment_datasets:
            ds = sentiment_datasets[aspect]
            t = len(ds["train"]) if "train" in ds else 0
            v = len(ds["val"]) if "val" in ds else 0
            te = len(ds["test"]) if "test" in ds else 0
            print(f"  {aspect:<28} {t:>6} {v:>6} {te:>6}")
        else:
            print(f"  {aspect:<28} {'—':>6} {'—':>6} {'—':>6}")

    print(f"\n{'─' * 70}")
    print(f"  Tokenizer: distilbert-base-uncased (max_length=128)")
    print(f"  Sentiment mapping: {SENTIMENT_MAP}")
    print(f"  Saved to: {hf_dataset_dir}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
