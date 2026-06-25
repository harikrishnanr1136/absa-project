"""
Result Card Component — full prediction result display.

Renders the complete output from ABSAInferencePipeline.predict() as a
structured card with overall sentiment, per-aspect details, and timing.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.utils.app_helpers import (
    format_aspect_name,
    get_overall_sentiment_label,
    format_inference_time,
)
from app.components.sentiment_badge import render_sentiment_badge
from app.components.confidence_bar import render_confidence_bar


def render_result_card(prediction: dict):
    """
    Render a complete result card for an ABSA prediction.

    Takes the full prediction dict from ABSAInferencePipeline.predict() and displays:
    - Overall sentiment at top with large badge
    - Divider
    - Per-aspect breakdown (sorted by confidence descending):
        - Aspect name (formatted)
        - Sentiment badge
        - Confidence bar
    - Info message if no aspects detected
    - Inference time at bottom in small grey text

    Args:
        prediction: Dict with keys: feedback, detected_aspects, aspect_sentiments,
                   confidence_scores, overall_sentiment, inference_time_ms
    """
    import streamlit as st

    detected_aspects = prediction.get("detected_aspects", [])
    aspect_sentiments = prediction.get("aspect_sentiments", {})
    confidence_scores = prediction.get("confidence_scores", {})
    overall_sentiment = prediction.get("overall_sentiment", "neutral")
    inference_time = prediction.get("inference_time_ms", 0.0)

    with st.container():
        # ── Overall Sentiment (top) ───────────────────────────────────────
        st.markdown("#### Overall Sentiment")
        render_sentiment_badge(overall_sentiment, size="large")
        st.markdown("")

        # Aggregated label if different from simple overall
        aggregated = get_overall_sentiment_label(aspect_sentiments)
        if aggregated != overall_sentiment and aggregated not in ("none detected",):
            st.caption(f"Aggregated: {aggregated}")

        st.divider()

        # ── Per-Aspect Breakdown ──────────────────────────────────────────
        if not detected_aspects:
            st.info("ℹ️ No specific telecom aspects detected in this feedback.")
        else:
            st.markdown(f"#### Detected Aspects ({len(detected_aspects)})")
            st.markdown("")

            # Sort aspects by confidence score descending
            sorted_aspects = sorted(
                detected_aspects,
                key=lambda a: confidence_scores.get(a, 0.0),
                reverse=True,
            )

            for aspect in sorted_aspects:
                sentiment = aspect_sentiments.get(aspect, "neutral")
                score = confidence_scores.get(aspect, 0.0)

                # Two-column layout: badge on left, bar on right
                col_badge, col_bar = st.columns([1, 3])

                with col_badge:
                    st.markdown(f"**{format_aspect_name(aspect)}**")
                    render_sentiment_badge(sentiment, size="small")

                with col_bar:
                    render_confidence_bar(aspect, score, sentiment)

                st.markdown("")  # Spacing between aspects

        # ── Inference Time (bottom) ───────────────────────────────────────
        st.markdown("")
        st.markdown(
            f'<p style="color: #999; font-size: 12px; text-align: right;">'
            f'⏱ Inference time: {format_inference_time(inference_time)}</p>',
            unsafe_allow_html=True,
        )
