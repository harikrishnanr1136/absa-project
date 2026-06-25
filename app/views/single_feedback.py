"""
Page 1: Single Feedback Analysis.

Allows user to enter one telecom customer feedback and receive
aspect detection + sentiment classification results.

No business logic here — only UI and calls to inference pipeline and components.
"""

import logging
import traceback

import streamlit as st

from app.components.result_card import render_result_card
from app.utils.app_helpers import get_sample_feedbacks, get_sentiment_emoji

logger = logging.getLogger(__name__)


# ─── Cached Model Loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load inference pipeline once per session."""
    from src.inference import ABSAInferencePipeline
    return ABSAInferencePipeline()


# ─── Public Entry Point ───────────────────────────────────────────────────────

def render_page():
    """Public entry point called by app.py page router."""
    render()


# ─── Page Render ──────────────────────────────────────────────────────────────

def render():
    """Render the single feedback analysis page."""

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1 — Header
    # ═══════════════════════════════════════════════════════════════════════

    st.title("📡 Customer Feedback Analyzer")
    st.markdown("Analyze telecom customer feedback for aspects and sentiments")
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2 — Input Area
    # ═══════════════════════════════════════════════════════════════════════

    # Sample feedback selector
    samples = get_sample_feedbacks()
    sample_options = ["-- Select a sample --"] + samples

    def on_sample_change():
        """Callback: populate text area when sample is selected."""
        selected = st.session_state.get("sample_selector", "-- Select a sample --")
        if selected != "-- Select a sample --":
            st.session_state["feedback_input"] = selected

    def on_clear():
        """Callback: clear text area and selector."""
        st.session_state["feedback_input"] = ""
        st.session_state["sample_selector"] = "-- Select a sample --"

    selected_sample = st.selectbox(
        "Or choose a sample feedback:",
        options=sample_options,
        key="sample_selector",
        on_change=on_sample_change,
    )

    # Text area for feedback input
    feedback = st.text_area(
        "Enter customer feedback",
        placeholder="e.g. The internet speed is great but customer support is very slow to respond...",
        height=120,
        max_chars=500,
        key="feedback_input",
    )

    # Action buttons
    col_analyze, col_clear = st.columns(2)

    with col_analyze:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with col_clear:
        st.button("🗑️ Clear", use_container_width=True, on_click=on_clear)

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3 — Results Area
    # ═══════════════════════════════════════════════════════════════════════

    if analyze_clicked:
        # Validation
        if not feedback or feedback.strip() == "":
            st.warning("⚠️ Please enter some feedback text before analyzing.")
            return

        # Load pipeline
        pipeline = load_pipeline()

        # Run inference with error handling
        try:
            with st.spinner("Analyzing feedback..."):
                prediction = pipeline.predict(feedback.strip())

            logger.info(f"Prediction complete: {len(prediction.get('detected_aspects', []))} aspects, "
                        f"overall={prediction.get('overall_sentiment')}")

            # Display result card
            st.markdown("---")
            render_result_card(prediction)

            # Expandable raw data
            with st.expander("📋 View raw prediction data"):
                st.json(prediction)

            # Add to history
            st.session_state["history"].insert(0, prediction)
            # Keep last 10 in memory
            st.session_state["history"] = st.session_state["history"][:10]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            logger.error(traceback.format_exc())
            st.error(f"❌ Analysis failed: {str(e)}")
            st.markdown("Please try again or check that models are loaded correctly.")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4 — History
    # ═══════════════════════════════════════════════════════════════════════

    history = st.session_state.get("history", [])

    if history:
        st.markdown("---")
        st.markdown("#### 📜 Analysis History")
        st.caption(f"Showing last {min(len(history), 5)} analyses")

        # Display last 5 as collapsed expanders
        for i, past_prediction in enumerate(history[:5]):
            past_feedback = past_prediction.get("feedback", "")
            past_overall = past_prediction.get("overall_sentiment", "neutral")
            emoji = get_sentiment_emoji(past_overall)

            # Truncate feedback for expander label
            label_text = past_feedback[:60] + ("..." if len(past_feedback) > 60 else "")
            expander_label = f"{emoji} {label_text} — {past_overall}"

            with st.expander(expander_label, expanded=False):
                render_result_card(past_prediction)

        # Clear history button
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state["history"] = []
            st.rerun()
