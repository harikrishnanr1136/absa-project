"""
Evaluation module for ABSA Telecom models.
Provides metrics computation and reporting utilities.
"""

import logging
from typing import List

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: List[str], y_pred: List[str], label_names: List[str] = None) -> dict:
    """
    Compute evaluation metrics for model predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Optional list of label names for the report

    Returns:
        Dict containing accuracy, macro_f1, weighted_f1, and classification_report
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    logger.info(f"Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


def print_evaluation_report(y_true: List[str], y_pred: List[str], label_names: List[str] = None):
    """Print formatted evaluation report."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(classification_report(y_true, y_pred, target_names=label_names))
    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logger.info(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    logger.info("=" * 60)
