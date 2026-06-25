"""
Text Preprocessing Pipeline for ABSA Telecom Dataset

Provides:
- PreprocessingPipeline class: consolidated end-to-end preprocessing
- Standalone functions: clean_text, tokenize, remove_stopwords, lemmatize

Pipeline order:
    clean_text → tokenize (abbreviation expansion) → remove_stopwords
    (keep negations + domain terms) → lemmatize (POS-aware) → join tokens
"""

import re
import ssl
import logging
import math
import os
from typing import List, Optional

import joblib
import nltk

# Bypass SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Default Configuration ────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "abbreviation_map": {
        "plz": "please",
        "ur": "your",
        "u": "you",
        "v": "very",
        "tbh": "to be honest",
        "ngl": "not going to lie",
        "asap": "as soon as possible",
        "omg": "oh my god",
        "cant": "cannot",
        "wont": "will not",
        "isnt": "is not",
        "sim": "sim",
        "4g": "4g",
        "5g": "5g",
    },
    "domain_terms": {
        "4g", "5g", "sim", "ott", "ivr", "apn", "imei",
        "volte", "wifi", "hotspot", "recharge", "prepaid", "postpaid",
    },
    # Words to keep even though they are stopwords — critical for sentiment analysis.
    # Negation words flip polarity ("not good" != "good"), intensifiers modify strength,
    # and contrastive conjunctions signal sentiment shifts.
    "sentiment_keep_words": {
        "not", "no", "never", "very", "too", "but", "however",
        "although", "though", "yet", "only", "just",
    },
}

# Module-level references for backward compatibility
ABBREVIATION_MAP = DEFAULT_CONFIG["abbreviation_map"]
DOMAIN_TERMS = DEFAULT_CONFIG["domain_terms"]
SENTIMENT_KEEP_WORDS = DEFAULT_CONFIG["sentiment_keep_words"]


# ═══════════════════════════════════════════════════════════════════════════════
# PreprocessingPipeline Class
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """
    Consolidated preprocessing pipeline for ABSA telecom feedback text.

    Pipeline order:
        1. clean_text: lowercase, remove URLs/HTML/punctuation, normalize whitespace
        2. tokenize: NLTK word_tokenize + abbreviation expansion
        3. remove_stopwords: NLTK stopwords minus sentiment-critical & domain terms
        4. lemmatize: POS-aware WordNet lemmatization (skip domain terms)
        5. join: rejoin tokens into a single cleaned string

    Usage:
        pipeline = PreprocessingPipeline()
        cleaned_texts = pipeline.fit_transform(train_texts)
        cleaned_val = pipeline.transform(val_texts)
        token_lists = pipeline.get_tokens(texts)
        pipeline.save("pipeline.joblib")
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: Optional dict with keys:
                - abbreviation_map: dict of abbreviation -> expansion
                - domain_terms: set of terms to never remove/modify
                - sentiment_keep_words: set of stopwords to preserve
        """
        config = config or DEFAULT_CONFIG

        self.abbreviation_map = config.get("abbreviation_map", DEFAULT_CONFIG["abbreviation_map"])
        self.domain_terms = set(config.get("domain_terms", DEFAULT_CONFIG["domain_terms"]))
        self.sentiment_keep_words = set(config.get("sentiment_keep_words", DEFAULT_CONFIG["sentiment_keep_words"]))

        # NLTK resources
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words("english"))
        self._effective_stopwords = self._stopwords - self.sentiment_keep_words - self.domain_terms

        # Fitted flag
        self._is_fitted = False

        logger.info("PreprocessingPipeline initialized")
        logger.info(f"  Abbreviations: {len(self.abbreviation_map)} entries")
        logger.info(f"  Domain terms: {len(self.domain_terms)} terms")
        logger.info(f"  Sentiment keep words: {len(self.sentiment_keep_words)} words")

    # ─── Core Pipeline Steps ──────────────────────────────────────────────

    def _clean_text(self, text) -> str:
        """Step 1: Clean raw text (lowercase, remove URLs/HTML/punctuation, normalize whitespace)."""
        if text is None:
            return ""
        if isinstance(text, float) and math.isnan(text):
            return ""
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^a-z0-9\s']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Step 2: Tokenize and expand abbreviations."""
        if not text:
            return []

        tokens = word_tokenize(text)

        expanded = []
        for token in tokens:
            if token in self.domain_terms:
                expanded.append(token)
            elif token in self.abbreviation_map:
                expansion = self.abbreviation_map[token]
                expanded.extend(expansion.split())
            else:
                expanded.append(token)

        return expanded

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Step 3: Remove stopwords, keep sentiment-critical and domain terms."""
        if not tokens:
            return []

        return [
            t for t in tokens
            if t in self.domain_terms
            or t in self.sentiment_keep_words
            or t not in self._effective_stopwords
        ]

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str):
        """Map Penn Treebank POS tag to WordNet POS constant."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Step 4: POS-aware lemmatization, skip domain terms."""
        if not tokens:
            return []

        tagged = pos_tag(tokens)
        lemmatized = []

        for token, tag in tagged:
            if token in self.domain_terms:
                lemmatized.append(token)
            else:
                wn_pos = self._get_wordnet_pos(tag)
                lemmatized.append(self._lemmatizer.lemmatize(token, pos=wn_pos))

        return lemmatized

    def _process_single(self, text) -> List[str]:
        """Run full pipeline on a single text, return token list."""
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return tokens

    # ─── Public Interface ─────────────────────────────────────────────────

    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Run full pipeline on training texts. Returns list of cleaned joined strings.

        Marks pipeline as fitted (abbreviation dict and domain terms are "learned"
        from the config and remain fixed for transform on val/test sets).

        Args:
            texts: List of raw feedback strings

        Returns:
            List of preprocessed strings (tokens joined by space)
        """
        logger.info(f"fit_transform: processing {len(texts)} texts")
        self._is_fitted = True

        results = []
        for text in texts:
            tokens = self._process_single(text)
            results.append(" ".join(tokens))

        logger.info(f"fit_transform: complete")
        return results

    def transform(self, texts: List[str]) -> List[str]:
        """
        Apply fitted pipeline to new texts (val/test sets).

        Uses the same abbreviation_map, domain_terms, and stopwords established
        during fit_transform — no re-fitting occurs.

        Args:
            texts: List of raw feedback strings

        Returns:
            List of preprocessed strings (tokens joined by space)
        """
        if not self._is_fitted:
            logger.warning("Pipeline not fitted yet. Call fit_transform() first or use transform() directly.")

        logger.info(f"transform: processing {len(texts)} texts")
        results = []
        for text in texts:
            tokens = self._process_single(text)
            results.append(" ".join(tokens))

        return results

    def get_tokens(self, texts: List[str]) -> List[List[str]]:
        """
        Return list of token lists (for embedding models like Word2Vec, FastText).

        Args:
            texts: List of raw feedback strings

        Returns:
            List of token lists (not joined)
        """
        logger.info(f"get_tokens: processing {len(texts)} texts")
        return [self._process_single(text) for text in texts]

    def save(self, path: str):
        """Save pipeline state with joblib."""
        state = {
            "abbreviation_map": self.abbreviation_map,
            "domain_terms": self.domain_terms,
            "sentiment_keep_words": self.sentiment_keep_words,
            "is_fitted": self._is_fitted,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Pipeline saved to: {path}")

    def load(self, path: str):
        """Load pipeline state from joblib file."""
        state = joblib.load(path)
        self.abbreviation_map = state["abbreviation_map"]
        self.domain_terms = set(state["domain_terms"])
        self.sentiment_keep_words = set(state["sentiment_keep_words"])
        self._is_fitted = state["is_fitted"]
        # Rebuild effective stopwords
        self._effective_stopwords = self._stopwords - self.sentiment_keep_words - self.domain_terms
        logger.info(f"Pipeline loaded from: {path}")

    def explain_pipeline(self, text: str):
        """
        Print step-by-step transformation trace for a single input text.
        Useful for debugging and interviews.

        Args:
            text: Raw input text to trace through the pipeline
        """
        print(f"\n{'═' * 70}")
        print(f"PIPELINE TRACE")
        print(f"{'═' * 70}")
        print(f"\n  INPUT: {repr(text)}")

        # Step 1: clean_text
        cleaned = self._clean_text(text)
        print(f"\n  STEP 1 - clean_text:")
        print(f"    → Lowercase, remove URLs/HTML/punctuation, normalize whitespace")
        print(f"    Result: {repr(cleaned)}")

        # Step 2: tokenize
        tokens = self._tokenize(cleaned)
        print(f"\n  STEP 2 - tokenize (with abbreviation expansion):")
        print(f"    → NLTK word_tokenize + expand abbreviations")
        print(f"    Tokens ({len(tokens)}): {tokens}")

        # Step 3: remove_stopwords
        filtered = self._remove_stopwords(tokens)
        removed = [t for t in tokens if t not in filtered]
        print(f"\n  STEP 3 - remove_stopwords:")
        print(f"    → Remove NLTK stopwords, KEEP negations + domain terms")
        print(f"    Kept ({len(filtered)}): {filtered}")
        print(f"    Removed ({len(removed)}): {removed}")

        # Step 4: lemmatize
        lemmatized = self._lemmatize(filtered)
        changes = [(f, l) for f, l in zip(filtered, lemmatized) if f != l]
        print(f"\n  STEP 4 - lemmatize (POS-aware):")
        print(f"    → WordNet lemmatizer with POS tags, skip domain terms")
        print(f"    Result ({len(lemmatized)}): {lemmatized}")
        if changes:
            print(f"    Changes: {', '.join(f'{f}→{l}' for f, l in changes)}")
        else:
            print(f"    Changes: none")

        # Step 5: join
        output = " ".join(lemmatized)
        print(f"\n  STEP 5 - join:")
        print(f"    → Rejoin tokens into string for TF-IDF/vectorizers")
        print(f"    Final: {repr(output)}")

        print(f"\n{'═' * 70}")
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Functions (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

# Instantiate a default pipeline for standalone function use
_default_pipeline = None


def _get_default_pipeline() -> PreprocessingPipeline:
    """Lazy-initialize the default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = PreprocessingPipeline()
    return _default_pipeline


def clean_text(text) -> str:
    """Clean raw feedback text. See PreprocessingPipeline._clean_text for details."""
    return _get_default_pipeline()._clean_text(text)


def tokenize(text) -> list:
    """Tokenize and expand abbreviations. See PreprocessingPipeline._tokenize."""
    pipeline = _get_default_pipeline()
    cleaned = pipeline._clean_text(text)
    return pipeline._tokenize(cleaned)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords keeping sentiment-critical and domain terms."""
    return _get_default_pipeline()._remove_stopwords(tokens)


def lemmatize(tokens: list) -> list:
    """POS-aware lemmatization, skip domain terms."""
    return _get_default_pipeline()._lemmatize(tokens)


def get_wordnet_pos(treebank_tag: str):
    """Map Penn Treebank POS tag to WordNet POS constant."""
    return PreprocessingPipeline._get_wordnet_pos(treebank_tag)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Suppress verbose logging for clean demo output
    logging.getLogger().setLevel(logging.WARNING)

    print("=" * 70)
    print("PreprocessingPipeline - DEMONSTRATION")
    print("=" * 70)

    # Initialize pipeline
    pipeline = PreprocessingPipeline()

    # ─── Test Cases ───────────────────────────────────────────────────────
    test_cases = [
        "plz fix ur 5g network asap!!! cant blv this is happening in 2024",
        "ngl the SIM activation was v fast. tbh impressed with the VoLTE quality",
        "Visit https://airtel.in for help. <b>Worst</b> customer support EVER 😤😤😤",
        "The OTT bundle with my recharge plan is amazing. Netflix and Prime included for just ₹599!",
        "omg billing is SO messed up. charged 200 extra for services I never subscribed to. custmer care useless",
    ]

    # ─── Explain Pipeline (detailed trace for first example) ──────────────
    print("\n")
    pipeline.explain_pipeline(test_cases[0])

    # ─── fit_transform demo ───────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("fit_transform() - Full Pipeline Results")
    print("=" * 70)

    results = pipeline.fit_transform(test_cases)

    for i, (original, processed) in enumerate(zip(test_cases, results), 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i}] INPUT:  {original}")
        print(f"      OUTPUT: {processed}")

    # ─── get_tokens demo ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("get_tokens() - Token Lists (for embeddings)")
    print("=" * 70)

    token_lists = pipeline.get_tokens(test_cases[:3])
    for i, (text, tokens) in enumerate(zip(test_cases[:3], token_lists), 1):
        print(f"\n  [{i}] {len(tokens)} tokens: {tokens}")

    # ─── transform demo (simulate val/test) ───────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("transform() - Apply to new texts (val/test)")
    print("=" * 70)

    new_texts = [
        "ur intrenet speed is terrible. v bad service tbh",
        "Happy with the prepaid recharge. Good value for money!",
    ]
    transformed = pipeline.transform(new_texts)
    for i, (orig, proc) in enumerate(zip(new_texts, transformed), 1):
        print(f"\n  [{i}] INPUT:  {orig}")
        print(f"      OUTPUT: {proc}")

    # ─── save/load demo ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("save() / load() - Pipeline Persistence")
    print("=" * 70)

    save_path = "/tmp/absa_pipeline_test.joblib"
    pipeline.save(save_path)
    print(f"\n  Saved to: {save_path}")

    # Load into a new instance
    new_pipeline = PreprocessingPipeline()
    new_pipeline.load(save_path)
    print(f"  Loaded from: {save_path}")

    # Verify same output
    verify = new_pipeline.transform(["plz fix ur 5g network"])
    print(f"  Verify: 'plz fix ur 5g network' -> '{verify[0]}'")

    print(f"\n{'=' * 70}")
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
