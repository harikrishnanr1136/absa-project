"""
DistilBERT Models for Telecom ABSA.

Provides:
  - AspectDetectionModel: Multi-label classification (15 aspects)
  - SentimentClassificationModel: Multi-class classification (3 sentiments)
  - count_parameters(): Utility to inspect model size

Architecture notes:
  Both models use the [CLS] token representation from DistilBERT as the sentence
  embedding, which is standard practice for classification tasks. The [CLS] token
  is a special token prepended to every input during BERT pretraining. During the
  Next Sentence Prediction (NSP) pretraining objective, the model learns to encode
  the entire sequence meaning into this single token's hidden state. This makes it
  a natural choice as a fixed-size sentence representation for downstream classification.
"""

import logging

import torch
import torch.nn as nn
from transformers import DistilBertModel

from src.config import load_config

logger = logging.getLogger(__name__)


class AspectDetectionModel(nn.Module):
    """
    Multi-label aspect detection using DistilBERT + linear head + sigmoid.

    Why sigmoid (not softmax)?
        In multi-label classification, multiple aspects can be present simultaneously
        in a single feedback. Sigmoid treats each of the 15 outputs independently,
        producing a probability in [0, 1] for each aspect. This allows predicting
        any combination of aspects (e.g., "network_coverage" AND "call_quality").
        Softmax would force outputs to sum to 1, implying mutual exclusivity —
        which is incorrect for multi-label ABSA where aspects co-occur frequently.

    Architecture:
        DistilBERT → [CLS] hidden state (768) → Dropout(0.3) → Linear(768→15) → Sigmoid
    """

    def __init__(self, num_labels: int = 15, dropout_prob: float = 0.3,
                 model_name: str = "distilbert-base-uncased"):
        super().__init__()

        logger.info(f"AspectDetectionModel: loading '{model_name}' (num_labels={num_labels})")

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)  # 768 → 15
        self.sigmoid = nn.Sigmoid()

        logger.info(f"AspectDetectionModel: initialized "
                    f"(hidden={self.distilbert.config.hidden_size}, labels={num_labels}, dropout={dropout_prob})")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)

        Returns:
            Logits after sigmoid, shape (batch_size, num_labels) with values in [0, 1]
        """
        # Pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Take the [CLS] token output (first token of the sequence)
        # outputs.last_hidden_state shape: (batch_size, seq_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Apply linear classifier head
        logits = self.classifier(cls_output)  # (batch_size, 15)

        # Apply sigmoid for independent per-label probabilities
        logits = self.sigmoid(logits)

        return logits


class SentimentClassificationModel(nn.Module):
    """
    Per-aspect sentiment classification using DistilBERT + linear head.

    Why no softmax in forward (using CrossEntropyLoss instead)?
        CrossEntropyLoss in PyTorch internally applies log-softmax before computing
        the negative log-likelihood loss. Applying softmax manually in forward()
        would lead to double-softmax (numerical instability). Instead, we return raw
        logits and let the loss function handle the normalization. During inference,
        we apply argmax on raw logits (or softmax if probabilities are needed).

    Why softmax/CrossEntropy (not sigmoid)?
        Sentiment classification is a mutually exclusive multi-class problem: each
        feedback-aspect pair has exactly ONE sentiment (positive, negative, or neutral).
        Softmax ensures the three class probabilities sum to 1, correctly modeling
        the "choose one" constraint. Sigmoid would incorrectly allow predicting
        multiple sentiments simultaneously for the same aspect.

    Architecture:
        DistilBERT → [CLS] hidden state (768) → Dropout(0.3) → Linear(768→3) → raw logits
    """

    def __init__(self, num_classes: int = 3, dropout_prob: float = 0.3,
                 model_name: str = "distilbert-base-uncased"):
        super().__init__()

        logger.info(f"SentimentClassificationModel: loading '{model_name}' (num_classes={num_classes})")

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)  # 768 → 3

        logger.info(f"SentimentClassificationModel: initialized "
                    f"(hidden={self.distilbert.config.hidden_size}, classes={num_classes}, dropout={dropout_prob})")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)

        Returns:
            Raw logits, shape (batch_size, num_classes)
            Note: No softmax applied — CrossEntropyLoss handles this internally.
        """
        # Pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Take the [CLS] token output (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Apply linear classifier head — returns raw logits
        logits = self.classifier(cls_output)  # (batch_size, 3)

        return logits


def count_parameters(model: nn.Module, model_name: str = "Model"):
    """
    Print parameter count breakdown for a PyTorch model.

    Shows:
    - Total parameters
    - Trainable parameters
    - Non-trainable (frozen) parameters
    - Estimated model size in MB (assuming float32)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Size estimate: each float32 param = 4 bytes
    size_mb = total_params * 4 / (1024 * 1024)

    print(f"\n  {'─' * 50}")
    print(f"  {model_name} — Parameter Summary")
    print(f"  {'─' * 50}")
    print(f"  Total parameters:         {total_params:>12,}")
    print(f"  Trainable parameters:     {trainable_params:>12,}")
    print(f"  Non-trainable parameters: {non_trainable_params:>12,}")
    print(f"  Estimated size (float32): {size_mb:>12.2f} MB")
    print(f"  {'─' * 50}")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "size_mb": round(size_mb, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Instantiate and inspect both models
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("DistilBERT Models — Architecture Summary")
    print("=" * 60)

    # Load config for aspect count
    config = load_config()
    num_aspects = len(config["labels"]["aspects"])
    num_sentiments = len(config["labels"]["sentiments"])

    # Aspect Detection Model
    print(f"\n[1] Aspect Detection Model (multi-label, {num_aspects} aspects)")
    aspect_model = AspectDetectionModel(num_labels=num_aspects)
    count_parameters(aspect_model, "AspectDetectionModel")

    # Sentiment Classification Model
    print(f"\n[2] Sentiment Classification Model (multi-class, {num_sentiments} sentiments)")
    sentiment_model = SentimentClassificationModel(num_classes=num_sentiments)
    count_parameters(sentiment_model, "SentimentClassificationModel")

    # Test forward pass
    print(f"\n[3] Forward Pass Test")
    dummy_input_ids = torch.randint(0, 30522, (2, 128))  # batch=2, seq_len=128
    dummy_attention_mask = torch.ones(2, 128, dtype=torch.long)

    with torch.no_grad():
        aspect_out = aspect_model(dummy_input_ids, dummy_attention_mask)
        sent_out = sentiment_model(dummy_input_ids, dummy_attention_mask)

    print(f"  Aspect output shape:    {aspect_out.shape} (values in [0,1])")
    print(f"  Aspect output sample:   {aspect_out[0, :5].tolist()}")
    print(f"  Sentiment output shape: {sent_out.shape} (raw logits)")
    print(f"  Sentiment output sample: {sent_out[0].tolist()}")

    print(f"\n{'=' * 60}")
    print("ALL MODELS VERIFIED")
    print("=" * 60)
