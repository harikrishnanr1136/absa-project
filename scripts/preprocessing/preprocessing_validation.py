"""
Preprocessing Validation Script for ABSA Telecom Dataset

Validates the entire preprocessing and feature extraction pipeline:
1. Loads cleaned CSV
2. Runs PreprocessingPipeline
3. Runs TFIDFFeatures
4. Runs SentenceEmbeddingFeatures
5. Validates all outputs
6. Saves artifacts to data/
"""

import logging
import os
import sys
import pickle

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "telecom_absa_cleaned.csv")

# Add scripts/preprocessing to path for imports
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import PreprocessingPipeline
from features import TFIDFFeatures, SentenceEmbeddingFeatures

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_check(name: str, condition: bool):
    """Print PASSED/FAILED for a validation check."""
    status = "✅ PASSED" if condition else "❌ FAILED"
    logger.info(f"  {status} — {name}")
    return condition


def main():
    logger.info("=" * 70)
    logger.info("PREPROCESSING VALIDATION PIPELINE")
    logger.info("=" * 70)

    all_passed = True

    # ─── Step 1: Load Data ────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 1: Loading data")
    logger.info("─" * 70)

    try:
        df = pd.read_csv(INPUT_PATH)
        n_samples = len(df)
        logger.info(f"  Loaded {n_samples} rows from {INPUT_PATH}")
    except Exception as e:
        logger.error(f"  Failed to load data: {e}")
        sys.exit(1)

    feedbacks = df["feedback"].tolist()

    # ─── Step 2: Preprocessing Pipeline ───────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 2: Running PreprocessingPipeline.fit_transform()")
    logger.info("─" * 70)

    pipeline = PreprocessingPipeline()
    preprocessed_texts = pipeline.fit_transform(feedbacks)
    logger.info(f"  Preprocessed {len(preprocessed_texts)} texts")

    # ─── Step 3: TF-IDF Features ─────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 3: Running TFIDFFeatures.fit_transform()")
    logger.info("─" * 70)

    tfidf = TFIDFFeatures()
    X_tfidf = tfidf.fit_transform(preprocessed_texts)
    logger.info(f"  TF-IDF matrix shape: {X_tfidf.shape}")

    # ─── Step 4: Sentence Embeddings ─────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 4: Running SentenceEmbeddingFeatures.fit_transform()")
    logger.info("─" * 70)

    try:
        embedder = SentenceEmbeddingFeatures()
        X_emb = embedder.fit_transform(preprocessed_texts)
        logger.info(f"  Embedding matrix shape: {X_emb.shape}")
        embeddings_available = True
    except ImportError as e:
        logger.warning(f"  Skipped embeddings: {e}")
        X_emb = None
        embeddings_available = False

    # ─── Step 5: Validation Checks ───────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 5: Validation Checks")
    logger.info("─" * 70)

    # Check 1: No empty strings after preprocessing
    empty_count = sum(1 for t in preprocessed_texts if t.strip() == "")
    check1 = validate_check(
        f"No empty strings after preprocessing (found {empty_count} empty)",
        empty_count == 0
    )
    all_passed &= check1

    # Check 2: TF-IDF matrix shape matches (n_samples, <=10000)
    tfidf_shape_ok = (X_tfidf.shape[0] == n_samples and X_tfidf.shape[1] <= 10000)
    check2 = validate_check(
        f"TF-IDF shape ({X_tfidf.shape[0]}, {X_tfidf.shape[1]}) — rows={n_samples}, features<=10000",
        tfidf_shape_ok
    )
    all_passed &= check2

    # Check 3: Embedding matrix shape matches (n_samples, 384)
    if embeddings_available:
        emb_shape_ok = (X_emb.shape[0] == n_samples and X_emb.shape[1] == 384)
        check3 = validate_check(
            f"Embedding shape ({X_emb.shape[0]}, {X_emb.shape[1]}) — expected ({n_samples}, 384)",
            emb_shape_ok
        )
        all_passed &= check3
    else:
        logger.info("  ⏭️  SKIPPED — Embedding shape check (sentence-transformers not available)")

    # Check 4: No NaN or Inf in TF-IDF
    tfidf_dense = X_tfidf.toarray()
    tfidf_nan = np.isnan(tfidf_dense).sum()
    tfidf_inf = np.isinf(tfidf_dense).sum()
    check4 = validate_check(
        f"No NaN/Inf in TF-IDF matrix (NaN={tfidf_nan}, Inf={tfidf_inf})",
        tfidf_nan == 0 and tfidf_inf == 0
    )
    all_passed &= check4

    # Check 5: No NaN or Inf in embeddings
    if embeddings_available:
        emb_nan = np.isnan(X_emb).sum()
        emb_inf = np.isinf(X_emb).sum()
        check5 = validate_check(
            f"No NaN/Inf in embedding matrix (NaN={emb_nan}, Inf={emb_inf})",
            emb_nan == 0 and emb_inf == 0
        )
        all_passed &= check5
    else:
        logger.info("  ⏭️  SKIPPED — Embedding NaN/Inf check")

    # Check 6: All TF-IDF values >= 0
    min_val = tfidf_dense.min()
    check6 = validate_check(
        f"All TF-IDF values >= 0 (min value = {min_val:.6f})",
        min_val >= 0
    )
    all_passed &= check6

    # ─── Step 6: Save Artifacts ──────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("STEP 6: Saving artifacts")
    logger.info("─" * 70)

    # Save preprocessed texts
    preprocessed_path = os.path.join(DATA_DIR, "preprocessed_feedback.pkl")
    with open(preprocessed_path, "wb") as f:
        pickle.dump(preprocessed_texts, f)
    logger.info(f"  Saved: {preprocessed_path}")

    # Save TF-IDF matrix
    tfidf_path = os.path.join(DATA_DIR, "tfidf_features.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(X_tfidf, f)
    logger.info(f"  Saved: {tfidf_path}")

    # Save embedding matrix
    if embeddings_available:
        emb_path = os.path.join(DATA_DIR, "embedding_features.npy")
        np.save(emb_path, X_emb)
        logger.info(f"  Saved: {emb_path}")
    else:
        logger.info("  ⏭️  SKIPPED — Embedding save (not available)")

    # ─── Final Summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    if all_passed:
        logger.info("🎉 ALL VALIDATION CHECKS PASSED")
    else:
        logger.info("⚠️  SOME CHECKS FAILED — review output above")
    logger.info("=" * 70)

    # Assert for CI/CD usage
    assert all_passed, "Validation failed — see log output for details"


if __name__ == "__main__":
    main()
