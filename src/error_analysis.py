"""
Error Analysis for Model 2 (DistilBERT) — Misclassification Extraction & Categorization.

Runs inference on test set, compares with ground truth, classifies each error
by type and category, and saves structured error records.
"""

import json
import logging
import os
import re
from collections import Counter
from typing import List

import pandas as pd

from src.config import load_config, resolve_path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Noise detection patterns
NOISE_PATTERNS = [
    r"\bplz\b", r"\bur\b", r"\bu\b", r"\bv\b", r"\btbh\b", r"\bngl\b",
    r"\basap\b", r"\bomg\b", r"\bcant\b", r"\bwont\b", r"\bisnt\b",
    r"\bintrenet\b", r"\bcustmer\b", r"\bprblm\b", r"\breacharg\b",
    r"\bamazng\b", r"\bfrustrting\b",
]


def detect_noise(text: str) -> bool:
    """Check if feedback contains abbreviations or spelling noise."""
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in NOISE_PATTERNS)


def classify_aspect_error(true_aspect: str, pred_aspect: str,
                           true_sent: str, pred_sent: str) -> str:
    """
    Classify an individual aspect-level error.

    Returns one of:
    - "false_positive_aspect": predicted but not in ground truth
    - "false_negative_aspect": in ground truth but not predicted
    - "sentiment_flip": correct aspect, sentiment completely wrong (pos↔neg)
    - "neutral_confusion": correct aspect, neutral confused with pos/neg
    - "multi_aspect_partial": catch-all for partial errors
    """
    if true_sent is None and pred_sent is not None:
        return "false_positive_aspect"
    if true_sent is not None and pred_sent is None:
        return "false_negative_aspect"
    if true_sent == pred_sent:
        return None  # Not an error

    # Both exist but different sentiments
    flip_pairs = {("positive", "negative"), ("negative", "positive")}
    if (true_sent, pred_sent) in flip_pairs:
        return "sentiment_flip"

    if "neutral" in (true_sent, pred_sent):
        return "neutral_confusion"

    return "multi_aspect_partial"


def classify_error_category(aspect_errors: list, true_aspects: list, pred_aspects: list) -> str:
    """
    Classify overall error category for a feedback entry.

    Returns one of:
    - "aspect_detection_error": wrong aspects detected
    - "sentiment_error": correct aspects, wrong sentiment
    - "both_errors": both aspect and sentiment wrong
    - "multi_aspect_challenge": 3+ true aspects with at least one wrong
    """
    has_aspect_error = any(
        e["error_type"] in ("false_positive_aspect", "false_negative_aspect")
        for e in aspect_errors
    )
    has_sentiment_error = any(
        e["error_type"] in ("sentiment_flip", "neutral_confusion", "multi_aspect_partial")
        for e in aspect_errors
    )

    if len(true_aspects) >= 3 and (has_aspect_error or has_sentiment_error):
        return "multi_aspect_challenge"
    elif has_aspect_error and has_sentiment_error:
        return "both_errors"
    elif has_aspect_error:
        return "aspect_detection_error"
    elif has_sentiment_error:
        return "sentiment_error"
    else:
        return "unknown"


def main():
    logger.info("=" * 70)
    logger.info("ERROR ANALYSIS — Model 2 (DistilBERT)")
    logger.info("=" * 70)

    # ─── Config & Paths ───────────────────────────────────────────────────
    config = load_config()
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    os.makedirs(output_dir, exist_ok=True)

    # ─── Load Test Data ───────────────────────────────────────────────────
    test_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_path)
    test_df["aspects"] = test_df["aspects"].apply(json.loads)
    test_df["aspect_sentiments"] = test_df["aspect_sentiments"].apply(json.loads)
    logger.info(f"Loaded test set: {len(test_df)} rows")

    # ─── Run Inference ────────────────────────────────────────────────────
    logger.info("Loading ABSAInferencePipeline...")
    from src.inference import ABSAInferencePipeline
    pipeline = ABSAInferencePipeline()

    logger.info("Running predict_batch on test set...")
    predictions = pipeline.predict_batch(test_df["feedback"].tolist())
    logger.info(f"Predictions complete: {len(predictions)} results")

    # ─── Step 1: Collect Misclassifications ───────────────────────────────
    logger.info("\nCollecting misclassifications...")
    error_records = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        pred = predictions[idx]
        true_aspects = row["aspects"]
        true_sentiments = row["aspect_sentiments"]
        pred_aspects = pred.get("detected_aspects", [])
        pred_sentiments = pred.get("aspect_sentiments", {})

        # Find all aspect-level errors
        aspect_errors = []

        # Check all true aspects
        all_aspects = set(true_aspects) | set(pred_aspects)

        for aspect in all_aspects:
            true_sent = true_sentiments.get(aspect)
            pred_sent = pred_sentiments.get(aspect)

            # Determine if this is an error
            if aspect in true_aspects and aspect not in pred_aspects:
                # False negative aspect
                aspect_errors.append({
                    "aspect": aspect,
                    "true_sentiment": true_sent,
                    "predicted_sentiment": None,
                    "error_type": "false_negative_aspect",
                })
            elif aspect not in true_aspects and aspect in pred_aspects:
                # False positive aspect
                aspect_errors.append({
                    "aspect": aspect,
                    "true_sentiment": None,
                    "predicted_sentiment": pred_sent,
                    "error_type": "false_positive_aspect",
                })
            elif aspect in true_aspects and aspect in pred_aspects:
                # Both have it — check sentiment
                if true_sent != pred_sent:
                    error_type = classify_aspect_error(aspect, aspect, true_sent, pred_sent)
                    if error_type:
                        aspect_errors.append({
                            "aspect": aspect,
                            "true_sentiment": true_sent,
                            "predicted_sentiment": pred_sent,
                            "error_type": error_type,
                        })

        # Skip if no errors
        if not aspect_errors:
            continue

        # Classify overall error category
        error_category = classify_error_category(aspect_errors, true_aspects, pred_aspects)
        feedback_text = row["feedback"]

        error_records.append({
            "id": int(row.get("id", idx)),
            "feedback": feedback_text,
            "feedback_length": len(feedback_text.split()),
            "true_aspects": true_aspects,
            "predicted_aspects": pred_aspects,
            "aspect_errors": aspect_errors,
            "error_category": error_category,
            "noise_present": detect_noise(feedback_text),
        })

    logger.info(f"Total misclassified rows: {len(error_records)} / {len(test_df)}")

    # ─── Step 2 & 3: Summarize Error Types and Categories ─────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ERROR DISTRIBUTION")
    logger.info("─" * 70)

    # Error type distribution
    error_type_counts = Counter()
    for record in error_records:
        for ae in record["aspect_errors"]:
            error_type_counts[ae["error_type"]] += 1

    print(f"\n  Error Types (individual aspect errors):")
    print(f"  {'─' * 50}")
    total_aspect_errors = sum(error_type_counts.values())
    for error_type, count in error_type_counts.most_common():
        pct = count * 100 / max(total_aspect_errors, 1)
        print(f"    {error_type:<30} {count:>4} ({pct:.1f}%)")
    print(f"  {'─' * 50}")
    print(f"    {'TOTAL':<30} {total_aspect_errors:>4}")

    # Error category distribution
    category_counts = Counter(r["error_category"] for r in error_records)

    print(f"\n  Error Categories (per feedback):")
    print(f"  {'─' * 50}")
    for cat, count in category_counts.most_common():
        pct = count * 100 / max(len(error_records), 1)
        print(f"    {cat:<30} {count:>4} ({pct:.1f}%)")
    print(f"  {'─' * 50}")
    print(f"    {'TOTAL MISCLASSIFIED':<30} {len(error_records):>4} / {len(test_df)}")

    # Noise correlation
    noise_errors = sum(1 for r in error_records if r["noise_present"])
    print(f"\n  Noise correlation:")
    print(f"    Errors with noise: {noise_errors} / {len(error_records)} "
          f"({noise_errors * 100 / max(len(error_records), 1):.1f}%)")

    # Length correlation
    if error_records:
        avg_error_len = sum(r["feedback_length"] for r in error_records) / len(error_records)
        avg_all_len = test_df["feedback"].apply(lambda x: len(str(x).split())).mean()
        print(f"    Avg length of errors: {avg_error_len:.1f} words (vs {avg_all_len:.1f} overall)")

    # ─── Save ─────────────────────────────────────────────────────────────
    output_data = {
        "total_test_samples": len(test_df),
        "total_misclassified": len(error_records),
        "error_rate": round(len(error_records) / len(test_df), 4),
        "error_type_distribution": dict(error_type_counts),
        "error_category_distribution": dict(category_counts),
        "noise_error_count": noise_errors,
        "records": error_records,
    }

    json_path = os.path.join(output_dir, "misclassifications.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_path}")

    logger.info("\n" + "=" * 70)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Error Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════

# Semantic similarity explanations for commonly confused aspect pairs
ASPECT_SIMILARITY_REASONS = {
    ("pricing", "value_for_money"): "Both relate to cost perception — pricing is the raw cost, value_for_money is cost relative to benefits.",
    ("pricing", "recharge_plans"): "Recharge plans inherently involve pricing; users often discuss both together.",
    ("recharge_plans", "value_for_money"): "Plan selection is driven by value assessment — the two concepts overlap heavily.",
    ("internet_speed", "5g_experience"): "5G is experienced primarily through speed improvements — feedback about one often implies the other.",
    ("network_coverage", "call_quality"): "Poor coverage directly causes poor call quality — users conflate cause and effect.",
    ("billing", "pricing"): "Billing reflects pricing decisions — unexpected charges feel like pricing issues.",
    ("data_balance", "data_validity"): "Both concern data availability — balance is how much, validity is how long.",
    ("mobile_app_experience", "data_balance"): "Users check data balance through the app — app bugs get reported as balance issues.",
    ("customer_support", "billing"): "Users contact support about billing — the two are often mentioned together.",
    ("ott_bundle_services", "recharge_plans"): "OTT bundles are packaged within recharge plans — hard to separate.",
}


def analyze_error_patterns():
    """
    Load misclassifications.json and generate a structured error pattern report.

    Analyses:
    1. Common failure cases (top 5 aspects by error rate)
    2. Confusing aspect pairs (15x15 confusion matrix)
    3. Sentiment misclassification patterns with examples
    4. Multi-aspect prediction challenges (error rate vs aspect count)
    5. Noise impact on error rate
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(eda_dir, exist_ok=True)

    # Load misclassifications
    misclass_path = os.path.join(output_dir, "misclassifications.json")
    with open(misclass_path, "r") as f:
        data = json.load(f)

    records = data["records"]
    total_test = data["total_test_samples"]

    logger.info("\n" + "=" * 70)
    logger.info("ERROR PATTERN ANALYSIS")
    logger.info("=" * 70)

    report = {}

    # ─── Analysis 1: Common Failure Cases ─────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ANALYSIS 1: Top 5 Aspects with Highest Misclassification Rate")
    logger.info("─" * 70)

    aspect_error_counts = Counter()
    aspect_total_counts = Counter()
    aspect_error_types = {}

    for record in records:
        for ae in record["aspect_errors"]:
            aspect = ae["aspect"]
            aspect_error_counts[aspect] += 1
            if aspect not in aspect_error_types:
                aspect_error_types[aspect] = Counter()
            aspect_error_types[aspect][ae["error_type"]] += 1

    # Load test.csv for total aspect counts
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df["aspects"] = test_df["aspects"].apply(json.loads)
    for _, row in test_df.iterrows():
        for a in row["aspects"]:
            aspect_total_counts[a] += 1

    # Compute error rates
    aspect_error_rates = {}
    for aspect in aspect_labels:
        total = aspect_total_counts.get(aspect, 0)
        errors = aspect_error_counts.get(aspect, 0)
        rate = errors / max(total, 1)
        aspect_error_rates[aspect] = {
            "total": total,
            "errors": errors,
            "rate": round(rate, 4),
            "most_common_error": aspect_error_types.get(aspect, Counter()).most_common(1)[0][0]
            if aspect in aspect_error_types else None,
        }

    top5 = sorted(aspect_error_rates.items(), key=lambda x: x[1]["rate"], reverse=True)[:5]

    print(f"\n  {'Aspect':<28} {'Errors':>7} {'Total':>7} {'Rate':>7} {'Common Error Type'}")
    print(f"  {'─' * 75}")
    for aspect, info in top5:
        print(f"  {aspect:<28} {info['errors']:>7} {info['total']:>7} "
              f"{info['rate']:>7.2%} {info['most_common_error'] or 'N/A'}")

    report["top5_failure_aspects"] = [
        {"aspect": a, **info} for a, info in top5
    ]

    # ─── Analysis 2: Confused Aspect Pairs ────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ANALYSIS 2: Most Confused Aspect Pairs")
    logger.info("─" * 70)

    # Build 15x15 confusion matrix for aspect detection
    n = len(aspect_labels)
    aspect_idx = {a: i for i, a in enumerate(aspect_labels)}
    aspect_cm = np.zeros((n, n), dtype=int)

    for record in records:
        true_set = set(record["true_aspects"])
        pred_set = set(record["predicted_aspects"])

        # False positives: predicted but not true
        for pred_a in pred_set - true_set:
            if pred_a in aspect_idx:
                # Find which true aspect was closest (first true aspect)
                for true_a in true_set:
                    if true_a in aspect_idx:
                        aspect_cm[aspect_idx[true_a]][aspect_idx[pred_a]] += 1

        # False negatives: true but not predicted
        for true_a in true_set - pred_set:
            if true_a in aspect_idx:
                # Missed entirely — mark as self if no prediction
                for pred_a in pred_set:
                    if pred_a in aspect_idx:
                        aspect_cm[aspect_idx[true_a]][aspect_idx[pred_a]] += 1

    # Top 5 confused pairs (off-diagonal)
    off_diag = aspect_cm.copy()
    np.fill_diagonal(off_diag, 0)

    confused_pairs = []
    flat_indices = np.argsort(off_diag.ravel())[::-1]
    seen = set()
    for flat_idx in flat_indices:
        i, j = divmod(flat_idx, n)
        if off_diag[i, j] == 0:
            break
        pair = tuple(sorted([aspect_labels[i], aspect_labels[j]]))
        if pair not in seen:
            seen.add(pair)
            reason = ASPECT_SIMILARITY_REASONS.get(pair, "Semantic overlap in telecom context.")
            confused_pairs.append({
                "aspect_a": aspect_labels[i],
                "aspect_b": aspect_labels[j],
                "count": int(off_diag[i, j]),
                "reason": reason,
            })
        if len(confused_pairs) >= 5:
            break

    print(f"\n  Top 5 confused aspect pairs:")
    print(f"  {'─' * 70}")
    for cp in confused_pairs:
        print(f"    {cp['aspect_a']:<25} ↔ {cp['aspect_b']:<25} ({cp['count']} times)")
        print(f"      Why: {cp['reason']}")

    report["confused_aspect_pairs"] = confused_pairs

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    short_labels = [a.replace("_", "\n")[:15] for a in aspect_labels]
    sns.heatmap(off_diag, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, linewidths=0.5)
    ax.set_title("Aspect Detection Confusion (Off-Diagonal)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Aspect")
    ax.set_ylabel("True Aspect")
    plt.tight_layout()
    heatmap_path = os.path.join(eda_dir, "aspect_detection_confusion_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"  Saved: {heatmap_path}")

    # ─── Analysis 3: Sentiment Misclassification Patterns ─────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ANALYSIS 3: Sentiment Misclassification Patterns")
    logger.info("─" * 70)

    patterns = {
        "positive_as_negative": [],
        "negative_as_neutral": [],
        "neutral_as_positive_or_negative": [],
    }

    for record in records:
        for ae in record["aspect_errors"]:
            t = ae.get("true_sentiment")
            p = ae.get("predicted_sentiment")
            if t == "positive" and p == "negative":
                patterns["positive_as_negative"].append(record["feedback"][:100])
            elif t == "negative" and p == "neutral":
                patterns["negative_as_neutral"].append(record["feedback"][:100])
            elif t == "neutral" and p in ("positive", "negative"):
                patterns["neutral_as_positive_or_negative"].append(record["feedback"][:100])

    pattern_explanations = {
        "positive_as_negative": (
            "Often caused by sarcasm or backhanded compliments where surface-level positive "
            "words mask underlying negativity. Also occurs with mixed-sentiment multi-aspect "
            "feedback where the model bleeds negativity from one aspect to another."
        ),
        "negative_as_neutral": (
            "Mild or understated negative feedback (e.g., 'could be better', 'not great') "
            "lacks strong negative signal words, making it appear neutral to the model. "
            "Common in formal complaint-style writing that avoids emotional language."
        ),
        "neutral_as_positive_or_negative": (
            "Factual statements that mention aspects without clear sentiment get misclassified "
            "because the model associates certain aspects (e.g., 'billing', 'support') with "
            "polarity based on training distribution rather than contextual meaning."
        ),
    }

    report["sentiment_patterns"] = {}
    for pattern_name, examples in patterns.items():
        print(f"\n  {pattern_name} ({len(examples)} cases):")
        for ex in examples[:5]:
            print(f"    • \"{ex}\"")
        print(f"  Explanation: {pattern_explanations[pattern_name]}")
        report["sentiment_patterns"][pattern_name] = {
            "count": len(examples),
            "examples": examples[:5],
            "explanation": pattern_explanations[pattern_name],
        }

    # ─── Analysis 4: Multi-Aspect Challenges ──────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ANALYSIS 4: Error Rate vs Aspect Count")
    logger.info("─" * 70)

    # Count errors by aspect count groups
    aspect_count_groups = {1: {"total": 0, "errors": 0},
                           2: {"total": 0, "errors": 0},
                           3: {"total": 0, "errors": 0}}  # 3 = 3+

    error_ids = set(r["id"] for r in records)

    for _, row in test_df.iterrows():
        n_aspects = len(row["aspects"])
        group = min(n_aspects, 3)
        aspect_count_groups[group]["total"] += 1
        if row.get("id", _) in error_ids or n_aspects in error_ids:
            aspect_count_groups[group]["errors"] += 1

    # Recount more accurately using feedback text matching
    error_feedbacks = set(r["feedback"] for r in records)
    for group in aspect_count_groups.values():
        group["errors"] = 0
    for _, row in test_df.iterrows():
        n_aspects = len(row["aspects"])
        group = min(n_aspects, 3)
        if row["feedback"] in error_feedbacks:
            aspect_count_groups[group]["errors"] += 1

    group_labels = ["1 aspect", "2 aspects", "3+ aspects"]
    error_rates = []
    for g, label in zip([1, 2, 3], group_labels):
        total = aspect_count_groups[g]["total"]
        errors = aspect_count_groups[g]["errors"]
        rate = errors / max(total, 1)
        error_rates.append(rate)
        print(f"  {label}: {errors}/{total} errors ({rate:.1%})")

    report["multi_aspect_error_rates"] = {
        label: {"total": aspect_count_groups[g]["total"],
                "errors": aspect_count_groups[g]["errors"],
                "rate": round(error_rates[i], 4)}
        for i, (g, label) in enumerate(zip([1, 2, 3], group_labels))
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(group_labels, error_rates, "o-", color="#E74C3C", linewidth=2, markersize=10)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_xlabel("Number of Aspects in Feedback", fontsize=12)
    ax.set_title("Error Rate vs Aspect Count", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for i, (label, rate) in enumerate(zip(group_labels, error_rates)):
        ax.annotate(f"{rate:.1%}", (i, rate), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(eda_dir, "error_rate_vs_aspect_count.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"  Saved: {plot_path}")

    # ─── Analysis 5: Noise Impact ─────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("ANALYSIS 5: Noise Impact on Error Rate")
    logger.info("─" * 70)

    noisy_total = sum(1 for _, row in test_df.iterrows() if detect_noise(row["feedback"]))
    clean_total = total_test - noisy_total
    noisy_errors = sum(1 for r in records if r["noise_present"])
    clean_errors = len(records) - noisy_errors

    noisy_rate = noisy_errors / max(noisy_total, 1)
    clean_rate = clean_errors / max(clean_total, 1)

    print(f"\n  Noisy feedback:  {noisy_errors}/{noisy_total} errors ({noisy_rate:.1%})")
    print(f"  Clean feedback:  {clean_errors}/{clean_total} errors ({clean_rate:.1%})")
    print(f"  Noise impact:    {'+' if noisy_rate > clean_rate else ''}"
          f"{(noisy_rate - clean_rate)*100:.1f} percentage points")

    report["noise_impact"] = {
        "noisy": {"total": noisy_total, "errors": noisy_errors, "rate": round(noisy_rate, 4)},
        "clean": {"total": clean_total, "errors": clean_errors, "rate": round(clean_rate, 4)},
        "difference_pp": round((noisy_rate - clean_rate) * 100, 2),
    }

    # ─── Save Report ──────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "error_pattern_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nSaved: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("ERROR PATTERN ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
    # Run pattern analysis if misclassifications exist
    output_dir = resolve_path(load_config()["outputs"]["models"])
    misclass_path = os.path.join(output_dir, "misclassifications.json")
    if os.path.exists(misclass_path):
        analyze_error_patterns()
