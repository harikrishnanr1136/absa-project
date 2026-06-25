"""
TF-IDF Feature Extraction for ABSA Telecom Dataset

Provides TFIDFFeatures class for converting preprocessed text into sparse feature matrices.

Why sublinear_tf=True?
    Customer feedback text often contains repeated words (e.g., "very very slow" or
    "bad bad service"). Without sublinear TF, raw term frequency gives disproportionate
    weight to these repetitions. sublinear_tf applies log-normalization: tf = 1 + log(tf),
    which dampens the effect of high raw counts. This ensures that a word appearing 10 times
    in a review doesn't get 10x the weight of a word appearing once — it captures *presence*
    and *moderate emphasis* rather than raw repetition, which is more informative for
    sentiment signals in short telecom feedback.

Why min_df=2?
    min_df=2 removes terms that appear in only one document. In customer feedback, single-
    occurrence terms are typically: misspelled words unique to one user (e.g., "netwrk"),
    random alphanumeric strings (order IDs, phone numbers), or ultra-rare slang that won't
    generalize. Removing these reduces feature dimensionality, prevents overfitting to
    individual user quirks, and focuses the model on terms that have at least minimal
    evidence across the corpus. This is a lightweight noise filter that preserves all
    meaningful vocabulary.
"""

import logging
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TFIDFFeatures:
    """
    TF-IDF feature extraction with analysis utilities for ABSA.

    Usage:
        tfidf = TFIDFFeatures()
        X_train = tfidf.fit_transform(train_texts)
        X_val = tfidf.transform(val_texts)
        tfidf.print_stats()
        top_features = tfidf.get_feature_names(y_train)
        tfidf.save("models/tfidf.joblib")
    """

    def __init__(self):
        """Initialize TF-IDF vectorizer with optimized parameters for telecom feedback."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,       # Cap vocabulary to top 10K terms by TF-IDF
            ngram_range=(1, 2),       # Unigrams + bigrams capture phrases like "not good", "call quality"
            sublinear_tf=True,        # Log-normalize term frequency (see module docstring)
            min_df=2,                 # Remove single-occurrence noise terms (see module docstring)
        )
        self._X_fitted = None  # Stores fitted matrix for stats
        self._is_fitted = False
        logger.info("TFIDFFeatures initialized (max_features=10000, ngrams=(1,2), sublinear_tf=True, min_df=2)")

    def fit_transform(self, texts: List[str]) -> "sparse matrix":
        """
        Fit the vectorizer on training texts and return the TF-IDF sparse matrix.

        Args:
            texts: List of preprocessed text strings (output of PreprocessingPipeline)

        Returns:
            Sparse CSR matrix of shape (n_samples, n_features)
        """
        logger.info(f"fit_transform: fitting on {len(texts)} documents")
        self._X_fitted = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info(f"fit_transform: complete — shape {self._X_fitted.shape}")
        return self._X_fitted

    def transform(self, texts: List[str]) -> "sparse matrix":
        """
        Transform new texts using the fitted vectorizer (for val/test sets).

        Args:
            texts: List of preprocessed text strings

        Returns:
            Sparse CSR matrix of shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit_transform() first.")

        logger.info(f"transform: transforming {len(texts)} documents")
        X = self.vectorizer.transform(texts)
        logger.info(f"transform: complete — shape {X.shape}")
        return X

    def get_feature_names(self, y_labels: Optional[List[str]] = None, top_n: int = 20) -> dict:
        """
        Get top features by mean TF-IDF score, optionally per sentiment class.

        If y_labels is provided, returns top features for each unique label.
        Otherwise, returns global top features by mean TF-IDF score.

        Args:
            y_labels: List of sentiment labels (same length as training data)
            top_n: Number of top features to return per class

        Returns:
            Dict: {label: [(feature, score), ...]} or {"global": [(feature, score), ...]}
        """
        if not self._is_fitted or self._X_fitted is None:
            raise RuntimeError("Vectorizer not fitted. Call fit_transform() first.")

        feature_names = self.vectorizer.get_feature_names_out()
        results = {}

        if y_labels is not None:
            # Per-class top features
            unique_labels = sorted(set(y_labels))
            y_array = np.array(y_labels)

            for label in unique_labels:
                mask = y_array == label
                class_matrix = self._X_fitted[mask]

                # Mean TF-IDF score per feature for this class
                mean_scores = np.asarray(class_matrix.mean(axis=0)).flatten()
                top_indices = mean_scores.argsort()[::-1][:top_n]

                results[label] = [
                    (feature_names[i], round(float(mean_scores[i]), 4))
                    for i in top_indices
                ]
        else:
            # Global top features
            mean_scores = np.asarray(self._X_fitted.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:top_n]

            results["global"] = [
                (feature_names[i], round(float(mean_scores[i]), 4))
                for i in top_indices
            ]

        return results

    def print_stats(self):
        """Print matrix shape, sparsity percentage, and memory size in MB."""
        if not self._is_fitted or self._X_fitted is None:
            print("  No fitted matrix available. Call fit_transform() first.")
            return

        X = self._X_fitted
        n_samples, n_features = X.shape
        n_nonzero = X.nnz
        total_elements = n_samples * n_features
        sparsity = (1 - n_nonzero / total_elements) * 100

        # Memory: data + indices + indptr for CSR
        memory_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        memory_mb = memory_bytes / (1024 * 1024)

        print(f"\n  {'─' * 50}")
        print(f"  TF-IDF Matrix Statistics")
        print(f"  {'─' * 50}")
        print(f"  Shape:          {n_samples} samples × {n_features} features")
        print(f"  Non-zero:       {n_nonzero:,} elements")
        print(f"  Total elements: {total_elements:,}")
        print(f"  Sparsity:       {sparsity:.2f}%")
        print(f"  Memory (CSR):   {memory_mb:.2f} MB")
        print(f"  Avg features/doc: {n_nonzero / n_samples:.1f}")
        print(f"  Vocabulary size:  {n_features}")
        print(f"  {'─' * 50}")

    def save(self, path: str):
        """Save fitted vectorizer to disk with joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"TFIDFFeatures saved to: {path}")

    def load(self, path: str):
        """Load a previously saved vectorizer from disk."""
        self.vectorizer = joblib.load(path)
        self._is_fitted = True
        logger.info(f"TFIDFFeatures loaded from: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence Embedding Features
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceEmbeddingFeatures:
    """
    Dense sentence embeddings using sentence-transformers (all-MiniLM-L6-v2).

    Produces 384-dimensional dense vectors that capture semantic meaning and context.
    Unlike TF-IDF, these embeddings understand word order, synonyms, and negation
    context — making them powerful for sentiment analysis where "not good" should be
    far from "good" in embedding space.

    Usage:
        embedder = SentenceEmbeddingFeatures()
        X_train = embedder.fit_transform(train_texts)
        X_val = embedder.transform(val_texts)
        embedder.print_stats()
        embedder.save("models/embeddings.npy")

    Requirements:
        pip install sentence-transformers torch
        (Requires Python ≤ 3.12 for PyTorch compatibility)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "  pip install sentence-transformers torch\n"
                "Note: Requires Python <= 3.12 for PyTorch compatibility."
            )

        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._embeddings = None
        logger.info(f"Model loaded — embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into dense embeddings. Returns numpy array of shape (n_samples, 384).

        This is stateless (no vocabulary fitting), but stores the result for stats.

        Args:
            texts: List of text strings (raw or preprocessed)

        Returns:
            numpy array of shape (n_samples, 384)
        """
        logger.info(f"fit_transform: encoding {len(texts)} texts")
        self._embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        )
        logger.info(f"fit_transform: complete — shape {self._embeddings.shape}")
        return self._embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into dense embeddings (stateless — same as fit_transform).

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, 384)
        """
        logger.info(f"transform: encoding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True,
        )
        logger.info(f"transform: complete — shape {embeddings.shape}")
        return embeddings

    def save(self, path: str):
        """Save encoded matrix as .npy file."""
        if self._embeddings is None:
            raise RuntimeError("No embeddings to save. Call fit_transform() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, self._embeddings)
        logger.info(f"Embeddings saved to: {path}")

    def load(self, path: str) -> np.ndarray:
        """Load matrix from .npy file."""
        self._embeddings = np.load(path)
        logger.info(f"Embeddings loaded from: {path} — shape {self._embeddings.shape}")
        return self._embeddings

    def print_stats(self):
        """Print shape, memory size in MB, average cosine similarity between random 100 sample pairs."""
        if self._embeddings is None:
            print("  No embeddings available. Call fit_transform() first.")
            return

        X = self._embeddings
        n_samples, n_dims = X.shape
        memory_mb = X.nbytes / (1024 * 1024)

        # Compute average cosine similarity between 100 random pairs
        rng = np.random.default_rng(42)
        n_pairs = min(100, n_samples * (n_samples - 1) // 2)
        idx_a = rng.integers(0, n_samples, size=n_pairs)
        idx_b = rng.integers(0, n_samples, size=n_pairs)
        # Avoid self-pairs
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        # Cosine similarity
        a_vecs = X[idx_a]
        b_vecs = X[idx_b]
        dot = np.sum(a_vecs * b_vecs, axis=1)
        norm_a = np.linalg.norm(a_vecs, axis=1)
        norm_b = np.linalg.norm(b_vecs, axis=1)
        cos_sim = dot / (norm_a * norm_b + 1e-8)
        avg_cos_sim = float(np.mean(cos_sim))

        print(f"\n  {'─' * 50}")
        print(f"  Sentence Embedding Statistics")
        print(f"  {'─' * 50}")
        print(f"  Model:            {self.model_name}")
        print(f"  Shape:            {n_samples} samples × {n_dims} dimensions")
        print(f"  Dtype:            {X.dtype}")
        print(f"  Memory:           {memory_mb:.2f} MB")
        print(f"  Dense:            Yes (no sparsity)")
        print(f"  Avg cosine sim:   {avg_cos_sim:.4f} (over {len(idx_a)} random pairs)")
        print(f"  {'─' * 50}")


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Function
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_tfidf(texts: List[str]):
    """
    Compare TF-IDF and Sentence Embedding representations on the same texts.

    Prints a comparison table covering dimensions, sparsity, context capture,
    training requirements, speed, and memory.

    Args:
        texts: List of preprocessed text strings
    """
    print(f"\n{'═' * 70}")
    print("COMPARISON: TF-IDF vs Sentence Embeddings")
    print(f"{'═' * 70}")
    print(f"\n  Encoding {len(texts)} texts with both methods...\n")

    # TF-IDF
    tfidf = TFIDFFeatures()
    X_tfidf = tfidf.fit_transform(texts)
    tfidf_dims = X_tfidf.shape[1]
    tfidf_memory = (X_tfidf.data.nbytes + X_tfidf.indices.nbytes + X_tfidf.indptr.nbytes) / (1024 * 1024)

    # Sentence Embeddings
    embedder = SentenceEmbeddingFeatures()
    X_emb = embedder.fit_transform(texts)
    emb_dims = X_emb.shape[1]
    emb_memory = X_emb.nbytes / (1024 * 1024)

    # Print comparison table
    print(f"  {'Property':<22} {'TF-IDF':<20} {'Sentence Embeddings':<25}")
    print(f"  {'─' * 67}")
    print(f"  {'Dimensions':<22} {tfidf_dims:<20} {emb_dims:<25}")
    print(f"  {'Sparse/Dense':<22} {'Sparse (CSR)':<20} {'Dense (float32)':<25}")
    print(f"  {'Captures context':<22} {'No':<20} {'Yes':<25}")
    print(f"  {'Training needed':<22} {'Yes (fit on corpus)':<20} {'No (pretrained)':<25}")
    print(f"  {'Inference speed':<22} {'Fast':<20} {'Slower (GPU helps)':<25}")
    print(f"  {'Memory (MB)':<22} {f'{tfidf_memory:.2f} MB':<20} {f'{emb_memory:.2f} MB':<25}")
    print(f"  {'Handles OOV words':<22} {'No (ignores)':<20} {'Yes (subword)':<25}")
    print(f"  {'Negation awareness':<22} {'Bigrams only':<20} {'Full context':<25}")
    print(f"  {'─' * 67}")

    print(f"\n  RECOMMENDATION:")
    print(f"  • Use TF-IDF for fast baselines, interpretable features, and low-resource settings.")
    print(f"  • Use Sentence Embeddings for higher accuracy on nuanced sentiment/aspect tasks.")
    print(f"  • Combine both (concatenate or ensemble) for best results in production ABSA.")
    print(f"{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main - Demonstration
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    DATA_PATH = os.path.join(PROJECT_DIR, "data", "absa_telecom_combined.csv")

    # Suppress verbose logs for demo
    logging.getLogger().setLevel(logging.WARNING)

    print("=" * 70)
    print("Feature Extraction - DEMONSTRATION")
    print("=" * 70)

    # Load data (subset for speed)
    import pandas as pd
    df = pd.read_csv(DATA_PATH).head(50)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(json.loads)

    # Preprocess
    from preprocessing import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    texts = pipeline.fit_transform(df["feedback"].tolist())

    # Extract dominant sentiment per entry as labels
    from collections import Counter
    labels = []
    for sent_dict in df["aspect_sentiments"]:
        counts = Counter(sent_dict.values())
        labels.append(counts.most_common(1)[0][0])

    # ═══════════════════════════════════════════════════════════════════════
    # TF-IDF Demo
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PART 1: TF-IDF Features")
    print("─" * 70)

    tfidf = TFIDFFeatures()
    X_tfidf = tfidf.fit_transform(texts)
    tfidf.print_stats()

    print("\n  Top 10 features per sentiment:")
    top_features = tfidf.get_feature_names(y_labels=labels, top_n=10)
    for sentiment, features in top_features.items():
        print(f"    {sentiment.upper()}: {', '.join(f[0] for f in features[:10])}")

    # ═══════════════════════════════════════════════════════════════════════
    # Sentence Embeddings Demo
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PART 2: Sentence Embeddings (all-MiniLM-L6-v2)")
    print("─" * 70)

    try:
        raw_texts = df["feedback"].tolist()[:10]  # Small subset for demo
        embedder = SentenceEmbeddingFeatures()
        X_emb = embedder.fit_transform(raw_texts)
        embedder.print_stats()

        # Save/Load test
        save_path = "/tmp/embeddings_test.npy"
        embedder.save(save_path)
        print(f"\n  Saved to: {save_path}")
        embedder.load(save_path)
        print(f"  Loaded and verified — shape: {embedder._embeddings.shape}")

        # Comparison
        print("\n" + "─" * 70)
        print("PART 3: Comparison")
        print("─" * 70)
        compare_with_tfidf(texts[:10])

    except ImportError as e:
        print(f"\n  ⚠️  Skipped: {e}")
        print("  Install: pip install sentence-transformers torch")

    except Exception as e:
        print(f"\n  ⚠️  Error during embedding: {e}")
        # Still print theoretical comparison
        tfidf_memory = (X_tfidf.data.nbytes + X_tfidf.indices.nbytes + X_tfidf.indptr.nbytes) / (1024**2)
        emb_memory_est = (len(texts) * 384 * 4) / (1024**2)

        print(f"\n  {'Property':<22} {'TF-IDF':<20} {'Sentence Embeddings':<25}")
        print(f"  {'─' * 67}")
        print(f"  {'Dimensions':<22} {X_tfidf.shape[1]:<20} {'384':<25}")
        print(f"  {'Sparse/Dense':<22} {'Sparse (CSR)':<20} {'Dense (float32)':<25}")
        print(f"  {'Captures context':<22} {'No':<20} {'Yes':<25}")
        print(f"  {'Training needed':<22} {'Yes (fit on corpus)':<20} {'No (pretrained)':<25}")
        print(f"  {'Inference speed':<22} {'Fast':<20} {'Slower (GPU helps)':<25}")
        print(f"  {'Memory (MB)':<22} {f'{tfidf_memory:.2f} MB':<20} {f'~{emb_memory_est:.2f} MB (est)':<25}")
        print(f"  {'Handles OOV words':<22} {'No (ignores)':<20} {'Yes (subword)':<25}")
        print(f"  {'Negation awareness':<22} {'Bigrams only':<20} {'Full context':<25}")
        print(f"  {'─' * 67}")

    print(f"\n{'=' * 70}")
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
