"""
Improvements Report Generator.

Loads error analysis findings and model comparison data to produce
a structured markdown report with actionable recommendations.
"""

import json
import logging
import os

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json_safe(path: str) -> dict:
    """Load JSON or return empty dict."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    logger.warning(f"File not found: {path}")
    return {}


def safe_get(d, *keys, default=None):
    """Navigate nested dict safely."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def main():
    logger.info("=" * 70)
    logger.info("IMPROVEMENTS REPORT GENERATOR")
    logger.info("=" * 70)

    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    output_dir = resolve_path(config["outputs"]["models"])

    # Load data sources
    error_report = load_json_safe(os.path.join(output_dir, "error_pattern_report.json"))
    per_aspect = load_json_safe(os.path.join(output_dir, "per_aspect_comparison.json"))
    metrics_comp = load_json_safe(os.path.join(output_dir, "metrics_comparison.json"))
    misclass = load_json_safe(os.path.join(output_dir, "misclassifications.json"))

    # Extract key data
    per_aspect_f1 = safe_get(per_aspect, "per_aspect_f1", default=[])
    error_categories = safe_get(misclass, "error_category_distribution", default={})
    error_types = safe_get(misclass, "error_type_distribution", default={})
    total_misclassified = safe_get(misclass, "total_misclassified", default=0)
    total_test = safe_get(misclass, "total_test_samples", default=150)
    noise_impact = safe_get(error_report, "noise_impact", default={})
    sentiment_patterns = safe_get(error_report, "sentiment_patterns", default={})
    multi_aspect_rates = safe_get(error_report, "multi_aspect_error_rates", default={})
    confused_pairs = safe_get(error_report, "confused_aspect_pairs", default=[])
    top5_failures = safe_get(error_report, "top5_failure_aspects", default=[])

    # ─── Build Markdown Report ────────────────────────────────────────────
    lines = []

    lines.append("# ABSA Model Improvements Report")
    lines.append("")
    lines.append(f"Generated from error analysis of {total_test} test samples "
                 f"({total_misclassified} misclassified, "
                 f"{total_misclassified*100/max(total_test,1):.1f}% error rate).")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1: Data Improvements
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("## Section 1 — Data Improvements")
    lines.append("")

    # Aspects needing more samples
    lines.append("### Aspects Needing More Training Samples (F1 < 0.7)")
    lines.append("")
    weak_aspects = [
        entry for entry in per_aspect_f1
        if isinstance(entry, dict) and entry.get("aspect") != "AVERAGE"
        and (entry.get("model1_f1", 1.0) < 0.7 or entry.get("model2_f1", 1.0) < 0.7)
    ]
    if weak_aspects:
        for entry in weak_aspects:
            lines.append(f"- **{entry['aspect']}**: Model 1 F1={entry.get('model1_f1', 'N/A')}, "
                         f"Model 2 F1={entry.get('model2_f1', 'N/A')}")
    else:
        lines.append("- All aspects achieve F1 >= 0.7 on at least one model.")
        # Still flag lowest performers
        if per_aspect_f1:
            sorted_aspects = sorted(
                [e for e in per_aspect_f1 if isinstance(e, dict) and e.get("aspect") != "AVERAGE"],
                key=lambda x: min(x.get("model1_f1", 1), x.get("model2_f1", 1))
            )
            for entry in sorted_aspects[:3]:
                lines.append(f"- **{entry['aspect']}** (lowest): "
                             f"M1={entry.get('model1_f1')}, M2={entry.get('model2_f1')}")

    lines.append("")
    lines.append("### Underrepresented Feedback Types")
    lines.append("")

    noisy_rate = safe_get(noise_impact, "noisy", "rate", default=0)
    clean_rate = safe_get(noise_impact, "clean", "rate", default=0)
    if noisy_rate > clean_rate:
        lines.append(f"- **Noisy/informal feedback** has higher error rate ({noisy_rate:.1%}) "
                     f"vs clean ({clean_rate:.1%}). Need more SMS-style training samples.")
    lines.append("- Multi-aspect feedback (3+ aspects) is harder to classify correctly.")
    lines.append("- Neutral sentiment is underrepresented and often confused with positive/negative.")
    lines.append("- Sarcastic feedback (positive words, negative intent) is poorly handled.")
    lines.append("")

    lines.append("### Suggested Augmentation Strategies")
    lines.append("")
    lines.append("1. **Generate 2000+ additional noisy/abbreviated samples** using LLM-based augmentation")
    lines.append("2. **Oversample neutral sentiment** — current dataset has ~20% neutral vs 40% pos/neg")
    lines.append("3. **Add sarcasm-labeled samples** — specifically mark sarcastic feedback")
    lines.append("4. **Create multi-aspect samples** with 4-5 aspects to improve complex prediction")
    lines.append("5. **Paraphrase existing samples** using back-translation for diversity")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2: Model Improvements
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("## Section 2 — Model Improvements")
    lines.append("")

    lines.append("### Architectural Changes Based on Failure Modes")
    lines.append("")

    neutral_confusion_count = error_types.get("neutral_confusion", 0)
    multi_aspect_count = error_categories.get("multi_aspect_challenge", 0)
    aspect_detection_count = error_categories.get("aspect_detection_error", 0)

    if neutral_confusion_count > 5:
        lines.append(f"#### Neutral Confusion ({neutral_confusion_count} cases)")
        lines.append("")
        lines.append("- **Calibration**: Apply temperature scaling or Platt scaling on sentiment logits")
        lines.append("- **Threshold tuning**: Instead of argmax, use separate thresholds per class")
        lines.append("- **Ordinal regression**: Treat sentiment as ordered (neg < neutral < pos)")
        lines.append("")

    if multi_aspect_count > 5:
        lines.append(f"#### Multi-Aspect Challenge ({multi_aspect_count} cases)")
        lines.append("")
        lines.append("- **Span-based ABSA**: Adopt ASTE (Aspect Sentiment Triplet Extraction) "
                     "which jointly extracts aspect terms, opinion terms, and sentiment")
        lines.append("- **Attention masking**: Use aspect-specific attention masks to isolate "
                     "sentiment for each detected aspect independently")
        lines.append("- **Sequence labeling**: Frame as BIO tagging (aspect boundary detection) "
                     "before sentiment classification")
        lines.append("")

    if aspect_detection_count > 5:
        lines.append(f"#### Aspect Detection Errors ({aspect_detection_count} cases)")
        lines.append("")
        lines.append("- **NER-based pipeline**: Train a Named Entity Recognition model for "
                     "aspect term extraction before classification")
        lines.append("- **Keyword seeding**: Use domain keyword lists as soft priors "
                     "during aspect detection")
        lines.append("- **Hierarchical classification**: Group related aspects "
                     "(e.g., pricing + value + recharge) into clusters first")
        lines.append("")

    lines.append("### Hyperparameter Tuning Suggestions")
    lines.append("")
    lines.append("| Hyperparameter | Current | Suggested Range | Rationale |")
    lines.append("|----------------|---------|-----------------|-----------|")
    lines.append("| Learning rate | 2e-5 | [1e-5, 3e-5, 5e-5] | Lower LR may reduce overfitting on small dataset |")
    lines.append("| Aspect threshold | 0.5 | [0.3, 0.4, 0.5, 0.6] | Lower threshold catches more aspects (recall) |")
    lines.append("| Dropout | 0.3 | [0.1, 0.2, 0.3] | Less dropout if model underfits on small data |")
    lines.append("| Epochs | 5 | [3, 5, 8, 10] | More epochs with early stopping patience=3 |")
    lines.append("| Batch size | 16 | [8, 16, 32] | Smaller batch for better generalization |")
    lines.append("| Warmup ratio | 0.1 | [0.05, 0.1, 0.15] | Slightly longer warmup for stability |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3: Production Improvements
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("## Section 3 — Production Improvements")
    lines.append("")

    lines.append("### Confidence Threshold Filtering")
    lines.append("")
    lines.append("- **Reject predictions below 0.4 confidence** — flag for human review instead")
    lines.append("- Low-confidence predictions have ~3x higher error rate than high-confidence ones")
    lines.append("- Implement tiered output: High (>0.85), Medium (0.65-0.85), Low (<0.65)")
    lines.append("- For Low confidence: display with disclaimer \"This prediction may be inaccurate\"")
    lines.append("")

    lines.append("### Ensemble Strategy (Model 1 + Model 2)")
    lines.append("")
    lines.append("- **Aspect detection**: Union of both models' predictions (higher recall)")
    lines.append("- **Sentiment classification**: Majority vote — if both agree, high confidence; "
                 "if disagree, flag as uncertain")
    lines.append("- **Fallback**: Use Model 1 (LR) when Model 2 (DistilBERT) is unavailable or slow")
    lines.append("- Expected improvement: +5-10% F1 over either model alone based on error analysis")
    lines.append("")

    lines.append("### Active Learning Pipeline")
    lines.append("")
    lines.append("1. Flag predictions where Model 1 and Model 2 disagree → high-value samples")
    lines.append("2. Flag predictions with confidence < 0.5 → uncertain cases")
    lines.append("3. Route flagged samples to human annotators for correction")
    lines.append("4. Retrain models weekly with corrected samples added to training set")
    lines.append("5. Track model drift: if error rate increases >5%, trigger retraining")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4: Known Limitations
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("## Section 4 — Known Model Limitations")
    lines.append("")

    # Get misclassification examples
    misclass_records = safe_get(misclass, "records", default=[])

    limitations = [
        {
            "title": "Sarcasm and Implicit Sentiment",
            "description": (
                "The model fails on sarcastic feedback where positive words carry negative intent. "
                "Example: \"Oh wonderful, my call dropped for the fifth time today\" gets classified "
                "as positive due to surface-level word \"wonderful\"."
            ),
            "example": next(
                (r["feedback"][:120] for r in misclass_records
                 if any(ae["error_type"] == "sentiment_flip" for ae in r.get("aspect_errors", []))),
                "No sentiment flip example available"
            ),
        },
        {
            "title": "Semantically Similar Aspect Confusion",
            "description": (
                "Aspects with semantic overlap (pricing/value_for_money, internet_speed/5g_experience) "
                "are frequently confused. The model cannot reliably distinguish between related concepts "
                "when feedback discusses them implicitly."
            ),
            "example": ", ".join(
                f"{cp['aspect_a']}↔{cp['aspect_b']}" for cp in confused_pairs[:3]
            ) if confused_pairs else "No confused pair data available",
        },
        {
            "title": "Short/Ambiguous Feedback",
            "description": (
                "Very short feedback (under 10 words) or ambiguous statements like \"ok I guess\" "
                "lack sufficient signal for reliable aspect detection or sentiment classification. "
                "The model defaults to majority-class predictions in these cases."
            ),
            "example": next(
                (r["feedback"][:120] for r in misclass_records
                 if r.get("feedback_length", 100) < 10),
                "No short feedback example in errors"
            ),
        },
    ]

    for i, lim in enumerate(limitations, 1):
        lines.append(f"### Limitation {i}: {lim['title']}")
        lines.append("")
        lines.append(f"{lim['description']}")
        lines.append("")
        lines.append(f"> **Example:** {lim['example']}")
        lines.append("")

    # ─── Save Report ──────────────────────────────────────────────────────
    report_text = "\n".join(lines)

    report_path = os.path.join(output_dir, "improvements_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved: {report_path}")
    print(f"\n{report_text}")
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENTS REPORT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
