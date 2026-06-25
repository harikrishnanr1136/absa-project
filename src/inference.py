"""
Unified Inference Pipeline for DistilBERT ABSA System.

Provides ABSAInferencePipeline class for end-to-end prediction:
    text → aspect detection → per-aspect sentiment → structured output

Designed to be shared between Streamlit app, FastAPI endpoints, and evaluation scripts.
"""

import logging
import os
import time
from collections import Counter
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer

from src.config import load_config, resolve_path
from src.dl_model import AspectDetectionModel, SentimentClassificationModel
from src.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)

SENTIMENT_ID_TO_LABEL = {0: "positive", 1: "negative", 2: "neutral"}


class ABSAInferencePipeline:
    """
    Unified inference pipeline for Aspect-Based Sentiment Analysis.

    Pipeline: preprocess → detect aspects → predict sentiment per aspect

    Usage:
        pipeline = ABSAInferencePipeline()
        result = pipeline.predict("terrible network coverage in my area")
        results = pipeline.predict_batch(["text1", "text2", ...])
    """

    def __init__(self, config_path: str = None):
        """
        Initialize pipeline by loading all models and components.

        Args:
            config_path: Path to config.yaml. Uses default if None.
        """
        logger.info("=" * 60)
        logger.info("ABSAInferencePipeline — Initializing")
        logger.info("=" * 60)

        total_start = time.time()

        # Load config
        self.config = load_config(config_path)
        self.aspect_labels = self.config["labels"]["aspects"]
        self.num_aspects = len(self.aspect_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models_dir = resolve_path(self.config["models"]["dir"])

        logger.info(f"Device: {self.device}")

        # ── Load Tokenizer ────────────────────────────────────────────────
        t0 = time.time()
        model_name = self.config["dl_training"]["model_name"]
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded: {model_name} ({time.time()-t0:.2f}s)")

        # ── Load Preprocessing Pipeline ───────────────────────────────────
        t0 = time.time()
        self.preprocessor = PreprocessingPipeline()
        pipeline_path = resolve_path(self.config["models"]["pipeline"])
        if os.path.exists(pipeline_path):
            self.preprocessor.load(pipeline_path)
        logger.info(f"PreprocessingPipeline loaded ({time.time()-t0:.2f}s)")

        # ── Load MultiLabelBinarizer ──────────────────────────────────────
        t0 = time.time()
        mlb_path = resolve_path(self.config["models"]["mlb"])
        self.mlb = joblib.load(mlb_path)
        logger.info(f"MultiLabelBinarizer loaded ({time.time()-t0:.2f}s)")

        # ── Load Aspect Detection Model ───────────────────────────────────
        t0 = time.time()
        self.aspect_model = AspectDetectionModel(
            num_labels=self.num_aspects, model_name=model_name
        ).to(self.device)

        aspect_model_path = resolve_path(self.config["models"]["aspect_model_distilbert"])
        if os.path.exists(aspect_model_path):
            state_dict = torch.load(aspect_model_path, map_location=self.device)
            self.aspect_model.load_state_dict(state_dict)
            logger.info(f"AspectDetectionModel loaded from checkpoint ({time.time()-t0:.2f}s)")
        else:
            logger.warning(f"Aspect model not found at {aspect_model_path} — using random weights")

        self.aspect_model.eval()

        # ── Load Sentiment Classification Models ──────────────────────────
        t0 = time.time()
        self.sentiment_models = {}
        sentiment_path = resolve_path(self.config["models"]["sentiment_model_distilbert"])

        if os.path.exists(sentiment_path):
            all_states = torch.load(sentiment_path, map_location=self.device)

            for aspect in self.aspect_labels:
                if aspect in all_states:
                    state = all_states[aspect]

                    # Handle fallback models (majority class)
                    if isinstance(state, dict) and state.get("type") == "fallback":
                        self.sentiment_models[aspect] = {
                            "type": "fallback",
                            "prediction": state["prediction"],
                        }
                    else:
                        model = SentimentClassificationModel(
                            num_classes=3, model_name=model_name
                        ).to(self.device)
                        model.load_state_dict(state)
                        model.eval()
                        self.sentiment_models[aspect] = {"type": "model", "model": model}

            logger.info(f"Sentiment models loaded: {len(self.sentiment_models)} aspects ({time.time()-t0:.2f}s)")
        else:
            logger.warning(f"Sentiment models not found at {sentiment_path}")

        total_load_time = time.time() - total_start
        logger.info(f"\nTotal initialization time: {total_load_time:.2f}s")
        logger.info("=" * 60)

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single text for model input.

        Args:
            text: Raw feedback string

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors (1, max_length)
        """
        max_length = self.config["dl_training"]["max_length"]

        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

    def predict_aspects(self, text: str) -> Dict:
        """
        Detect aspects mentioned in the feedback.

        Args:
            text: Raw feedback string

        Returns:
            Dict with:
                - detected_aspects: list of aspect names above threshold
                - confidence_scores: dict of aspect -> confidence (float)
        """
        inputs = self.preprocess(text)

        with torch.no_grad():
            logits = self.aspect_model(inputs["input_ids"], inputs["attention_mask"])

        # logits already have sigmoid applied (model in eval mode)
        probs = logits.squeeze(0).cpu().numpy()

        # Apply threshold
        detected = []
        confidence_scores = {}

        for i, aspect in enumerate(self.aspect_labels):
            score = float(probs[i])
            confidence_scores[aspect] = round(score, 4)
            if score >= 0.5:
                detected.append(aspect)

        return {
            "detected_aspects": detected,
            "confidence_scores": confidence_scores,
        }

    def predict_sentiment(self, text: str, aspect: str) -> Dict:
        """
        Predict sentiment for a specific aspect in the feedback.

        Args:
            text: Raw feedback string
            aspect: Target aspect name

        Returns:
            Dict with:
                - sentiment: predicted label string
                - confidence: float probability
        """
        if aspect not in self.sentiment_models:
            # Default to neutral if no model available
            return {"sentiment": "neutral", "confidence": 0.0}

        model_info = self.sentiment_models[aspect]

        # Handle fallback models
        if model_info["type"] == "fallback":
            label_id = model_info["prediction"]
            return {
                "sentiment": SENTIMENT_ID_TO_LABEL[label_id],
                "confidence": 1.0,  # Fallback always returns same prediction
            }

        # Neural model prediction
        model = model_info["model"]
        inputs = self.preprocess(text)

        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])

        # Apply softmax for probabilities
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        return {
            "sentiment": SENTIMENT_ID_TO_LABEL[pred_id],
            "confidence": round(confidence, 4),
        }

    def predict(self, text: str) -> Dict:
        """
        Run full ABSA pipeline on a single text.

        Pipeline: preprocess → detect aspects → predict sentiment per aspect

        Args:
            text: Raw feedback string

        Returns:
            Dict with:
                - feedback: original text
                - detected_aspects: list of detected aspect names
                - aspect_sentiments: {aspect: sentiment_label}
                - confidence_scores: {aspect: confidence_float}
                - overall_sentiment: majority vote sentiment
                - inference_time_ms: total time in milliseconds
        """
        start = time.time()

        # Step 1: Detect aspects
        aspect_result = self.predict_aspects(text)
        detected_aspects = aspect_result["detected_aspects"]
        aspect_confidences = aspect_result["confidence_scores"]

        # Step 2: Predict sentiment for each detected aspect
        aspect_sentiments = {}
        sentiment_confidences = {}

        for aspect in detected_aspects:
            sent_result = self.predict_sentiment(text, aspect)
            aspect_sentiments[aspect] = sent_result["sentiment"]
            sentiment_confidences[aspect] = sent_result["confidence"]

        # Step 3: Overall sentiment (majority vote)
        if aspect_sentiments:
            sentiment_counts = Counter(aspect_sentiments.values())
            overall_sentiment = sentiment_counts.most_common(1)[0][0]
        else:
            overall_sentiment = "neutral"

        inference_time_ms = (time.time() - start) * 1000

        return {
            "feedback": text,
            "detected_aspects": detected_aspects,
            "aspect_sentiments": aspect_sentiments,
            "confidence_scores": sentiment_confidences,
            "aspect_detection_scores": {a: aspect_confidences[a] for a in detected_aspects},
            "overall_sentiment": overall_sentiment,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Run full pipeline on a batch of texts.

        Args:
            texts: List of raw feedback strings

        Returns:
            List of prediction dicts
        """
        start = time.time()
        results = [self.predict(text) for text in texts]
        total_time_ms = (time.time() - start) * 1000
        per_sample_ms = total_time_ms / max(len(texts), 1)

        logger.info(f"predict_batch: {len(texts)} texts in {total_time_ms:.1f}ms "
                    f"({per_sample_ms:.2f}ms/sample)")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("ABSAInferencePipeline — DEMO")
    print("=" * 70)

    # Initialize pipeline
    pipeline = ABSAInferencePipeline()

    # Test cases
    test_texts = [
        "The 5G speed is amazing but customer support is terrible when I call about billing issues.",
        "plz fix ur network coverage in my area. calls keep dropping every day.",
        "Really happy with the recharge plan value. OTT bundle with Netflix included is great!",
    ]

    print(f"\n{'─' * 70}")
    print("SINGLE PREDICTIONS")
    print(f"{'─' * 70}")

    for i, text in enumerate(test_texts, 1):
        result = pipeline.predict(text)
        print(f"\n  [{i}] Input: {text[:70]}...")
        print(f"      Aspects: {result['detected_aspects']}")
        print(f"      Sentiments: {result['aspect_sentiments']}")
        print(f"      Overall: {result['overall_sentiment']}")
        print(f"      Time: {result['inference_time_ms']:.2f}ms")

    print(f"\n{'─' * 70}")
    print("BATCH PREDICTION")
    print(f"{'─' * 70}")

    batch_results = pipeline.predict_batch(test_texts)
    print(f"\n  Processed {len(batch_results)} texts")
    for r in batch_results:
        print(f"    → {len(r['detected_aspects'])} aspects, overall={r['overall_sentiment']}, "
              f"time={r['inference_time_ms']:.1f}ms")

    print(f"\n{'=' * 70}")
    print("DEMO COMPLETE")
    print("=" * 70)
