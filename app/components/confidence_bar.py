"""
Confidence Bar Component — labeled progress bar for aspect confidence display.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils.app_helpers import (
    format_aspect_name,
    get_sentiment_color,
    get_confidence_level,
)


def render_confidence_bar(aspect: str, score: float, sentiment: str):
    """
    Render a labeled progress bar showing confidence for an aspect-sentiment pair.

    Displays:
        "Internet Speed — Positive (High confidence: 87%)"
        [████████████████████░░░░]

    Args:
        aspect: Snake_case aspect name (e.g., "internet_speed")
        score: Confidence score between 0.0 and 1.0
        sentiment: Sentiment label for color coding
    """
    import streamlit as st

    display_name = format_aspect_name(aspect)
    color = get_sentiment_color(sentiment)
    level = get_confidence_level(score)
    pct = int(score * 100)

    # Label above the bar
    label = f"**{display_name}** — {sentiment.capitalize()} ({level} confidence: {pct}%)"
    st.markdown(label)

    # Progress bar (st.progress only supports 0-100 int or 0.0-1.0 float)
    st.progress(score)

    # Colored underline accent via CSS
    st.markdown(
        f"""<div style="
            height: 3px;
            width: {pct}%;
            background-color: {color};
            border-radius: 2px;
            margin-top: -10px;
            margin-bottom: 8px;
        "></div>""",
        unsafe_allow_html=True,
    )
