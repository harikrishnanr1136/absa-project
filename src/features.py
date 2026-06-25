"""
Feature extraction module for ABSA Telecom project.

Provides two feature engineering approaches:
1. TFIDFFeatures: Sparse bag-of-words with TF-IDF weighting
2. SentenceEmbeddingFeatures: Dense embeddings via sentence-transformers

Also provides compare_features() for side-by-side comparison.
"""

import logging
import os
from typing import List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_config

logger = logging.getLogger(__name__)


class TFIDFFeatures:
    """
    TF-IDF feature extraction optimized for telecom customer feedback.

    Uses TfidfVectorizer with:
        max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._X_fitted = None
        self._is_fitted = False
        logger.info("TFIDFFeatures initialized (max_features=10000, ngram_range=(1,2))")

    def fit_transform(self, texts: List[str]):
        """
        Fit TfidfVectorizer on texts and return sparse TF-IDF matrix.

        Args:
            texts: List of preprocessed text strings

        Returns:
            Sparse CSR matrix of shape (n_samples, n_features)
        """
        logger.info(f"TFIDFFeatures.fit_transform: {len(texts)} documents")
        self._X_fitted = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info(f"TFIDFFeatures.fit_transform: output shape {self._X_fitted.shape}")
        return self._X_fitted

    def transform(self, texts: List[str]):
        """
        Transform texts using fitted vectorizer (for val/test sets).

        Args:
            texts: List of preprocessed text strings

        Returns:
            Sparse CSR matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit_transform() first.")
        logger.info(f"TFIDFFeatures.transform: {len(texts)} documents")
        X = self.vectorizer.transform(texts)
        logger.info(f"TFIDFFeatures.transform: output shape {X.shape}")
        return X

    def save(self, path: str):
        """Save fitted vectorizer to disk with joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"TFIDFFeatures saved to: {path}")

    def load(self, path: str):
        """Load previously saved vectorizer from disk."""
        self.vectorizer = joblib.load(path)
        self._is_fitted = True
        logger.info(f"TFIDFFeatures loaded from: {path}")

    def get_feature_names(self, y_labels: Optional[List[str]] = None, top_n: int = 20) -> dict:
        """Get top features by mean TF-IDF score, optionally per sentiment class."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        feature_names = self.vectorizer.get_feature_names_out()
        results = {}
        if y_labels is not None:
            y_array = np.array(y_labels)
            for label in sorted(set(y_labels)):
                mask = y_array == label
                mean_scores = np.asarray(self._X_fitted[mask].mean(axis=0)).flatten()
                top_idx = mean_scores.argsort()[::-1][:top_n]
                results[label] = [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]
        else:
            mean_scores = np.asarray(self._X_fitted.mean(axis=0)).flatten()
            top_idx = mean_scores.argsort()[::-1][:top_n]
            results["global"] = [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]
        return results


class SentenceEmbeddingFeatures:
    """
    Dense sentence embeddings using sentence-transformers.
    Model name loaded from config.yaml (default: all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = None):
        """
        Initialize with sentence-transformers model.

        Args:
            model_name: HuggingFace model ID. If None, loads from config.yaml.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers torch"
            )

        # Load model name from config if not provided
        if model_name is None:
            try:
                config = load_config()
                model_name = config.get("features", {}).get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")
            except Exception:
                model_name = "all-MiniLM-L6-v2"

        logger.info(f"SentenceEmbeddingFeatures: loading model '{model_name}'")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._embeddings = None
        logger.info(f"SentenceEmbeddingFeatures: model loaded — dim={self.model.get_sentence_embedding_dimension()}")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into dense embeddings. Returns numpy array (n_samples, 384).

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 384)
        """
        logger.info(f"SentenceEmbeddingFeatures.fit_transform: {len(texts)} texts")
        self._embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
        )
        logger.info(f"SentenceEmbeddingFeatures.fit_transform: output shape {self._embeddings.shape}")
        return self._embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts (stateless — no fitting needed for pretrained embeddings).

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 384)
        """
        logger.info(f"SentenceEmbeddingFeatures.transform: {len(texts)} texts")
        embeddings = self.model.encode(
            texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True
        )
        logger.info(f"SentenceEmbeddingFeatures.transform: output shape {embeddings.shape}")
        return embeddings

    def save(self, path: str):
        """Save encoded matrix as .npy file."""
        if self._embeddings is None:
            raise RuntimeError("No embeddings to save. Call fit_transform() first.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, self._embeddings)
        logger.info(f"SentenceEmbeddingFeatures saved to: {path}")

    def load(self, path: str) -> np.ndarray:
        """Load matrix from .npy file."""
        self._embeddings = np.load(path)
        logger.info(f"SentenceEmbeddingFeatures loaded from: {path} — shape {self._embeddings.shape}")
        return self._embeddings


def compare_features(texts: List[str]):
    """
    Compare TF-IDF and Sentence Embedding representations side-by-side.

    Prints:
    - TF-IDF matrix shape and sparsity
    - Embedding matrix shape and memory size in MB

    Args:
        texts: List of preprocessed text strings
    """
    logger.info(f"\n{'═' * 60}")
    logger.info("FEATURE COMPARISON: TF-IDF vs Sentence Embeddings")
    logger.info(f"{'═' * 60}")
    logger.info(f"\n  Input: {len(texts)} texts\n")

    # TF-IDF
    logger.info("compare_features: computing TF-IDF")
    tfidf = TFIDFFeatures()
    X_tfidf = tfidf.fit_transform(texts)

    n_samples, n_features = X_tfidf.shape
    n_nonzero = X_tfidf.nnz
    total_elements = n_samples * n_features
    sparsity = (1 - n_nonzero / total_elements) * 100
    tfidf_memory = (X_tfidf.data.nbytes + X_tfidf.indices.nbytes + X_tfidf.indptr.nbytes) / (1024 * 1024)

    logger.info(f"  TF-IDF:")
    logger.info(f"    Shape:      {X_tfidf.shape}")
    logger.info(f"    Sparsity:   {sparsity:.2f}%")
    logger.info(f"    Non-zero:   {n_nonzero:,} / {total_elements:,}")
    logger.info(f"    Memory:     {tfidf_memory:.3f} MB")

    # Embeddings
    try:
        logger.info("compare_features: computing sentence embeddings")
        embedder = SentenceEmbeddingFeatures()
        X_emb = embedder.fit_transform(texts)

        emb_memory = X_emb.nbytes / (1024 * 1024)

        logger.info(f"\n  Sentence Embeddings:")
        logger.info(f"    Shape:      {X_emb.shape}")
        logger.info(f"    Sparsity:   0.00% (dense)")
        logger.info(f"    Memory:     {emb_memory:.3f} MB")
        logger.info(f"    Dtype:      {X_emb.dtype}")

    except ImportError:
        emb_memory = (len(texts) * 384 * 4) / (1024 * 1024)
        logger.info(f"\n  Sentence Embeddings (estimated — library not available):")
        logger.info(f"    Shape:      ({len(texts)}, 384)")
        logger.info(f"    Sparsity:   0.00% (dense)")
        logger.info(f"    Memory:     ~{emb_memory:.3f} MB")

    logger.info(f"\n{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("src/features.py — TEST")
    print("=" * 60)

    # Sample preprocessed texts
    texts = [
        "please fix 5g network soon possible cannot blv intrenet speed bad",
        "not go lie sim activation very fast honest impress volte call quality",
        "oh god billing ever charge 200 extra service never subscribe care useless",
        "excellent network coverage call quality clear happy service overall",
        "data balance drain fast not match actual usage terrible experience",
    ]

    # TF-IDF test
    print("\n[1] TF-IDF Features:")
    tfidf = TFIDFFeatures()
    X = tfidf.fit_transform(texts)
    print(f"    Matrix shape: {X.shape}")
    print(f"    Sparsity: {(1 - X.nnz / (X.shape[0]*X.shape[1])) * 100:.1f}%")

    # Transform test
    X_new = tfidf.transform(["network coverage terrible area"])
    print(f"    Transform new text: {X_new.shape}")

    # Save/load test
    tfidf.save("/tmp/test_tfidf.joblib")
    tfidf2 = TFIDFFeatures()
    tfidf2.load("/tmp/test_tfidf.joblib")
    print(f"    Save/Load: OK")

    # Sentence Embedding test
    print("\n[2] Sentence Embedding Features:")
    try:
        embedder = SentenceEmbeddingFeatures()
        X_emb = embedder.fit_transform(texts)
        print(f"    Matrix shape: {X_emb.shape}")
        print(f"    Memory: {X_emb.nbytes / 1024:.1f} KB")

        X_new_emb = embedder.transform(["network coverage terrible area"])
        print(f"    Transform new text: {X_new_emb.shape}")

        embedder.save("/tmp/test_embeddings.npy")
        print(f"    Save: OK")
    except ImportError as e:
        print(f"    Skipped: {e}")

    # Comparison
    print("\n[3] Compare Features:")
    compare_features(texts)

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
