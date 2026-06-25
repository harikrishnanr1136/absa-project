"""
Sentiment Badge Component — colored pill/badge for sentiment display.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils.app_helpers import get_sentiment_color, get_sentiment_emoji


def render_sentiment_badge(sentiment: str, size: str = "medium"):
    """
    Render a colored pill/badge for a sentiment label using st.markdown with inline HTML/CSS.

    Args:
        sentiment: One of "positive", "negative", "neutral"
        size: "small" (12px), "medium" (14px), or "large" (18px)

    Renders:
        A styled inline badge like: 😊 Positive
    """
    import streamlit as st

    color = get_sentiment_color(sentiment)
    emoji = get_sentiment_emoji(sentiment)

    font_sizes = {"small": "12px", "medium": "14px", "large": "18px"}
    paddings = {"small": "3px 8px", "medium": "5px 12px", "large": "7px 16px"}

    font_size = font_sizes.get(size, "14px")
    padding = paddings.get(size, "5px 12px")

    badge_html = f"""
    <span style="
        display: inline-block;
        background-color: {color};
        color: white;
        padding: {padding};
        border-radius: 20px;
        font-size: {font_size};
        font-weight: 600;
        letter-spacing: 0.3px;
    ">{emoji} {sentiment.capitalize()}</span>
    """

    st.markdown(badge_html, unsafe_allow_html=True)
