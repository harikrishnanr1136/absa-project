"""
Preprocessing module for ABSA Telecom project.
Provides PreprocessingPipeline class with step-by-step text preprocessing.
"""

import re
import ssl
import os
import math
import logging
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
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logger = logging.getLogger(__name__)

# ─── Default Configuration ────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "abbreviation_map": {
        "plz": "please", "ur": "your", "u": "you", "v": "very",
        "tbh": "to be honest", "ngl": "not going to lie",
        "asap": "as soon as possible", "omg": "oh my god",
        "cant": "cannot", "wont": "will not", "isnt": "is not",
        "sim": "sim", "4g": "4g", "5g": "5g",
    },
    "domain_terms": {
        "4g", "5g", "sim", "ott", "ivr", "apn", "imei",
        "volte", "wifi", "hotspot", "recharge", "prepaid", "postpaid",
    },
    "sentiment_keep_words": {
        "not", "no", "never", "very", "too", "but", "however",
        "although", "though", "yet", "only", "just",
    },
}


class PreprocessingPipeline:
    """
    End-to-end text preprocessing pipeline for ABSA.

    Methods:
        clean_text(text) → lowercased, URLs/special chars removed, whitespace stripped
        tokenize(text) → word tokens with abbreviation expansion
        remove_stopwords(tokens) → filtered token list (keeps negations + domain terms)
        lemmatize(tokens) → POS-aware lemmatized token list
        fit_transform(texts) → full pipeline on training texts, returns joined strings
        transform(texts) → same pipeline on val/test texts using fitted state
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or DEFAULT_CONFIG
        self.abbreviation_map = config.get("abbreviation_map", DEFAULT_CONFIG["abbreviation_map"])
        self.domain_terms = set(config.get("domain_terms", DEFAULT_CONFIG["domain_terms"]))
        self.sentiment_keep_words = set(config.get("sentiment_keep_words", DEFAULT_CONFIG["sentiment_keep_words"]))
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words("english"))
        self._effective_stopwords = self._stopwords - self.sentiment_keep_words - self.domain_terms
        self._is_fitted = False
        logger.info("PreprocessingPipeline initialized")

    def clean_text(self, text) -> str:
        """
        Clean raw text: lowercase, remove URLs, remove special characters, strip whitespace.

        Args:
            text: Raw input string (or None/NaN)

        Returns:
            Cleaned string
        """
        try:
            if text is None:
                logger.info("clean_text: received None, returning empty string")
                return ""
            if isinstance(text, float) and math.isnan(text):
                logger.info("clean_text: received NaN, returning empty string")
                return ""
            if not isinstance(text, str):
                text = str(text)

            input_len = len(text)
            text = text.lower()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"[^a-z0-9\s']", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            logger.info(f"clean_text: input_chars={input_len} → output_chars={len(text)}")
            return text

        except Exception as e:
            logger.error(f"clean_text: failed on input '{str(text)[:50]}...' — {e}")
            raise ValueError(f"clean_text failed: {e}") from e

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK word_tokenize with abbreviation expansion.

        Args:
            text: Cleaned text string

        Returns:
            List of tokens with abbreviations expanded
        """
        try:
            if not text:
                logger.info("tokenize: empty input, returning empty list")
                return []

            tokens = word_tokenize(text)
            input_count = len(tokens)

            expanded = []
            for token in tokens:
                if token in self.domain_terms:
                    expanded.append(token)
                elif token in self.abbreviation_map:
                    expanded.extend(self.abbreviation_map[token].split())
                else:
                    expanded.append(token)

            logger.info(f"tokenize: input_tokens={input_count} → output_tokens={len(expanded)}")
            return expanded

        except Exception as e:
            logger.error(f"tokenize: failed on input '{str(text)[:50]}...' — {e}")
            raise ValueError(f"tokenize failed: {e}") from e

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove English stopwords while keeping sentiment-critical and domain terms.

        Preserves: negation words (not, no, never), intensifiers (very, too),
        contrastive conjunctions (but, however), and all domain terms.

        Args:
            tokens: List of token strings

        Returns:
            Filtered list with stopwords removed
        """
        try:
            if not tokens:
                logger.info("remove_stopwords: empty input, returning empty list")
                return []

            input_count = len(tokens)
            filtered = [
                t for t in tokens
                if t in self.domain_terms
                or t in self.sentiment_keep_words
                or t not in self._effective_stopwords
            ]

            logger.info(f"remove_stopwords: input_tokens={input_count} → output_tokens={len(filtered)} "
                        f"(removed {input_count - len(filtered)})")
            return filtered

        except Exception as e:
            logger.error(f"remove_stopwords: failed — {e}")
            raise ValueError(f"remove_stopwords failed: {e}") from e

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using NLTK WordNetLemmatizer with POS-aware processing.

        Skips domain terms (returned as-is). Uses POS tags to select correct
        lemmatization form (noun/verb/adjective/adverb).

        Args:
            tokens: List of token strings

        Returns:
            List of lemmatized tokens
        """
        try:
            if not tokens:
                logger.info("lemmatize: empty input, returning empty list")
                return []

            input_count = len(tokens)
            tagged = pos_tag(tokens)

            lemmatized = []
            for token, tag in tagged:
                if token in self.domain_terms:
                    lemmatized.append(token)
                else:
                    wn_pos = self._get_wordnet_pos(tag)
                    lemmatized.append(self._lemmatizer.lemmatize(token, pos=wn_pos))

            changes = sum(1 for a, b in zip(tokens, lemmatized) if a != b)
            logger.info(f"lemmatize: input_tokens={input_count} → output_tokens={len(lemmatized)} "
                        f"({changes} tokens modified)")
            return lemmatized

        except Exception as e:
            logger.error(f"lemmatize: failed — {e}")
            raise ValueError(f"lemmatize failed: {e}") from e

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

    def _process_single(self, text) -> List[str]:
        """Run all pipeline steps on a single text and return token list."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Run full pipeline on training texts. Returns list of cleaned joined strings.

        Marks the pipeline as fitted — abbreviation_map and domain_terms are
        "learned" from config and remain fixed for subsequent transform() calls.

        Args:
            texts: List of raw feedback strings

        Returns:
            List of preprocessed strings (tokens joined by space)
        """
        try:
            logger.info(f"fit_transform: processing {len(texts)} texts")
            self._is_fitted = True
            results = [" ".join(self._process_single(t)) for t in texts]
            non_empty = sum(1 for r in results if r.strip())
            logger.info(f"fit_transform: complete — {len(results)} outputs, {non_empty} non-empty")
            return results

        except Exception as e:
            logger.error(f"fit_transform: failed — {e}")
            raise RuntimeError(f"fit_transform failed: {e}") from e

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
        try:
            if not self._is_fitted:
                logger.warning("transform: pipeline not fitted yet — proceeding anyway")

            logger.info(f"transform: processing {len(texts)} texts")
            results = [" ".join(self._process_single(t)) for t in texts]
            logger.info(f"transform: complete — {len(results)} outputs")
            return results

        except Exception as e:
            logger.error(f"transform: failed — {e}")
            raise RuntimeError(f"transform failed: {e}") from e

    def get_tokens(self, texts: List[str]) -> List[List[str]]:
        """Return list of token lists (for embedding models like Word2Vec)."""
        return [self._process_single(t) for t in texts]

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
        self._effective_stopwords = self._stopwords - self.sentiment_keep_words - self.domain_terms
        logger.info(f"Pipeline loaded from: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Test with 3 telecom feedback samples
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = PreprocessingPipeline()

    test_cases = [
        "plz fix ur 5g network asap!!! cant blv the intrenet speed is SO bad in my area. visit https://help.airtel.in for zero help 😤",
        "ngl the SIM activation was v fast and customer support was helpful. tbh impressed with the VoLTE call quality on this prepaid plan.",
        "omg WORST billing ever!! charged 200 extra for services I never subscribed to. custmer care useless, kept me on hold for 30 mins.",
    ]

    print("=" * 70)
    print("PreprocessingPipeline — TEST WITH TELECOM FEEDBACK")
    print("=" * 70)

    # Test individual methods step-by-step on first sample
    print(f"\n{'─' * 70}")
    print("STEP-BY-STEP TRACE (Sample 1):")
    print(f"{'─' * 70}")
    print(f"\n  RAW: {test_cases[0]}")

    cleaned = pipeline.clean_text(test_cases[0])
    print(f"  CLEAN: {cleaned}")

    tokens = pipeline.tokenize(cleaned)
    print(f"  TOKENIZE ({len(tokens)}): {tokens}")

    filtered = pipeline.remove_stopwords(tokens)
    print(f"  STOPWORDS ({len(filtered)}): {filtered}")

    lemmatized = pipeline.lemmatize(filtered)
    print(f"  LEMMATIZE ({len(lemmatized)}): {lemmatized}")

    print(f"  FINAL: {' '.join(lemmatized)}")

    # Test fit_transform on all 3 samples
    print(f"\n{'─' * 70}")
    print("fit_transform() — ALL 3 SAMPLES:")
    print(f"{'─' * 70}")

    results = pipeline.fit_transform(test_cases)
    for i, (orig, proc) in enumerate(zip(test_cases, results), 1):
        print(f"\n  [{i}] INPUT:  {orig[:80]}...")
        print(f"      OUTPUT: {proc}")

    # Test transform (simulated val/test)
    print(f"\n{'─' * 70}")
    print("transform() — NEW TEXTS (val/test):")
    print(f"{'─' * 70}")

    new_texts = ["ur data balance is wrong. v bad experience tbh"]
    val_results = pipeline.transform(new_texts)
    print(f"\n  INPUT:  {new_texts[0]}")
    print(f"  OUTPUT: {val_results[0]}")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)
