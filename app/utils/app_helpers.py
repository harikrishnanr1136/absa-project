"""
Utility functions for the Streamlit ABSA app.

Provides color mapping, formatting, sentiment aggregation, and sample data
for the telecom Aspect-Based Sentiment Analysis interface.
"""

from collections import Counter
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sentiment Color Mapping
# ═══════════════════════════════════════════════════════════════════════════════

def get_sentiment_color(sentiment: str) -> str:
    """
    Get hex color code for a sentiment label.

    Args:
        sentiment: One of "positive", "negative", "neutral"

    Returns:
        Hex color string:
            - "positive" → "#2ECC71" (green)
            - "negative" → "#E74C3C" (red)
            - "neutral"  → "#F39C12" (amber)
            - unknown    → "#95A5A6" (grey)
    """
    color_map = {
        "positive": "#2ECC71",
        "negative": "#E74C3C",
        "neutral": "#F39C12",
    }
    return color_map.get(sentiment.lower(), "#95A5A6")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sentiment Emoji Mapping
# ═══════════════════════════════════════════════════════════════════════════════

def get_sentiment_emoji(sentiment: str) -> str:
    """
    Get emoji for a sentiment label.

    Args:
        sentiment: One of "positive", "negative", "neutral"

    Returns:
        Emoji string:
            - "positive" → "😊"
            - "negative" → "😞"
            - "neutral"  → "😐"
    """
    emoji_map = {
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐",
    }
    return emoji_map.get(sentiment.lower(), "❓")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Aspect Name Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_aspect_name(aspect: str) -> str:
    """
    Convert snake_case aspect name to human-readable Title Case.

    Handles special telecom abbreviations (5G, OTT, SIM) that should stay uppercase.

    Args:
        aspect: Snake_case aspect string (e.g., "internet_speed")

    Returns:
        Formatted string (e.g., "Internet Speed")

    Examples:
        "internet_speed"      → "Internet Speed"
        "5g_experience"       → "5G Experience"
        "ott_bundle_services" → "OTT Bundle Services"
        "sim_activation"      → "SIM Activation"
    """
    # Special abbreviations that should remain uppercase
    abbreviations = {"5g": "5G", "ott": "OTT", "sim": "SIM", "ivr": "IVR", "apn": "APN"}

    words = aspect.split("_")
    formatted = []
    for word in words:
        if word.lower() in abbreviations:
            formatted.append(abbreviations[word.lower()])
        else:
            formatted.append(word.capitalize())

    return " ".join(formatted)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Overall Sentiment Label
# ═══════════════════════════════════════════════════════════════════════════════

def get_overall_sentiment_label(aspect_sentiments: Dict[str, str]) -> str:
    """
    Determine overall sentiment label from per-aspect sentiments using majority vote.

    Args:
        aspect_sentiments: Dict mapping aspect names to sentiment labels
                          e.g., {"network_coverage": "negative", "call_quality": "positive"}

    Returns:
        Aggregated label:
            - If all same: that sentiment (e.g., "positive")
            - If positive > negative: "mostly positive"
            - If negative > positive: "mostly negative"
            - If tied or mixed: "mixed"
            - If empty dict: "none detected"
    """
    if not aspect_sentiments:
        return "none detected"

    counts = Counter(aspect_sentiments.values())
    unique_sentiments = set(aspect_sentiments.values())

    # All same
    if len(unique_sentiments) == 1:
        return list(unique_sentiments)[0]

    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)

    if pos > neg:
        return "mostly positive"
    elif neg > pos:
        return "mostly negative"
    else:
        return "mixed"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Confidence Level
# ═══════════════════════════════════════════════════════════════════════════════

def get_confidence_level(score: float) -> str:
    """
    Categorize a confidence score into human-readable level.

    Args:
        score: Float between 0 and 1

    Returns:
        - score >= 0.85: "High"
        - score >= 0.65: "Medium"
        - score <  0.65: "Low"
    """
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    else:
        return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Inference Time Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_inference_time(ms: float) -> str:
    """
    Format inference time in milliseconds to a human-readable string.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted string: "23ms" or "1.2s"
    """
    if ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms / 1000:.1f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Sample Feedbacks
# ═══════════════════════════════════════════════════════════════════════════════

def get_sample_feedbacks() -> List[str]:
    """
    Return a list of 8 realistic sample telecom feedback strings.

    Covers:
    - Different aspects (network, billing, support, 5G, OTT, etc.)
    - Different sentiments (positive, negative, neutral, mixed)
    - Different lengths (short, medium, long)
    - At least one multi-aspect feedback
    - At least one noisy/SMS-style feedback

    Returns:
        List of 8 sample feedback strings
    """
    return [
        # Positive, single aspect (5G)
        "5G speeds in my area are incredible. Consistently getting 600+ Mbps.",

        # Negative, single aspect (network)
        "Network coverage is terrible in my building. No signal on any floor.",

        # Multi-aspect, mixed sentiment (positive internet + negative billing)
        "The internet speed is excellent but billing is very confusing with hidden charges every month.",

        # Positive, customer support
        "Customer support resolved my issue in 5 minutes. Very impressed with the quick response.",

        # Negative, multiple aspects (recharge + value)
        "Recharge plans keep getting worse. Less data, shorter validity, higher price. Zero value for money.",

        # Neutral / ambiguous
        "It works I guess. Nothing special about the service.",

        # Noisy / SMS-style abbreviations
        "plz fix ur network coverage asap. cant even make calls in my area. v bad experience tbh",

        # Long, multi-aspect, mixed
        "The OTT bundle with Netflix and Prime is great value, and the mobile app works smoothly for recharges. However roaming charges are way too high compared to competitors.",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Test all functions
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("app_helpers.py — FUNCTION TESTS")
    print("=" * 60)

    # 1. get_sentiment_color
    print("\n1. get_sentiment_color:")
    for s in ["positive", "negative", "neutral", "unknown"]:
        print(f"   {s:<12} → {get_sentiment_color(s)}")

    # 2. get_sentiment_emoji
    print("\n2. get_sentiment_emoji:")
    for s in ["positive", "negative", "neutral"]:
        print(f"   {s:<12} → {get_sentiment_emoji(s)}")

    # 3. format_aspect_name
    print("\n3. format_aspect_name:")
    test_aspects = [
        "internet_speed", "5g_experience", "ott_bundle_services",
        "sim_activation", "network_coverage", "call_quality",
    ]
    for a in test_aspects:
        print(f"   {a:<25} → {format_aspect_name(a)}")

    # 4. get_overall_sentiment_label
    print("\n4. get_overall_sentiment_label:")
    test_cases = [
        {"network_coverage": "positive", "call_quality": "positive"},
        {"network_coverage": "negative", "billing": "negative"},
        {"internet_speed": "positive", "billing": "negative"},
        {"5g": "positive", "network": "negative", "call": "neutral"},
        {},
    ]
    for tc in test_cases:
        print(f"   {tc} → {get_overall_sentiment_label(tc)}")

    # 5. get_confidence_level
    print("\n5. get_confidence_level:")
    for score in [0.92, 0.75, 0.50, 0.85, 0.65, 0.64]:
        print(f"   {score:.2f} → {get_confidence_level(score)}")

    # 6. format_inference_time
    print("\n6. format_inference_time:")
    for ms in [23.5, 150.0, 1500.0, 0.5, 3200.0]:
        print(f"   {ms:.1f}ms → {format_inference_time(ms)}")

    # 7. get_sample_feedbacks
    print("\n7. get_sample_feedbacks:")
    samples = get_sample_feedbacks()
    print(f"   Count: {len(samples)}")
    for i, s in enumerate(samples, 1):
        print(f"   [{i}] {s[:60]}{'...' if len(s)>60 else ''}")

    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETE")
    print("=" * 60)
