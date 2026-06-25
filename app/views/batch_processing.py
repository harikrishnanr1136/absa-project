"""
Page 2: Batch CSV Processing.

Allows user to upload a CSV file with multiple customer feedbacks,
run batch inference, and view dashboard analytics on the results.

No business logic here — delegates to inference pipeline, csv_helpers,
batch_runner, and dashboard_charts.
"""

import io
import logging
import traceback

import pandas as pd
import streamlit as st

from app.utils.csv_helpers import (
    validate_csv,
    clean_csv_for_inference,
    prepare_results_dataframe,
    results_to_download_csv,
    get_batch_summary_stats,
)
from app.utils.batch_runner import run_batch_inference, estimate_batch_time
from app.utils.dashboard_charts import (
    sentiment_distribution_pie,
    aspect_frequency_bar,
    positive_negative_trend_bar,
    aspect_sentiment_heatmap,
    confidence_distribution_histogram,
    top_negative_aspects_bar,
    feedback_length_vs_aspect_count_scatter,
)
from app.components.metric_cards import (
    render_summary_metrics_row,
    render_top_insights,
    render_error_summary,
)

logger = logging.getLogger(__name__)


# ─── Cached Model Loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load inference pipeline once per session."""
    from src.inference import ABSAInferencePipeline
    return ABSAInferencePipeline()


# ─── Sample Template ──────────────────────────────────────────────────────────

def generate_sample_csv() -> bytes:
    """Generate a sample CSV template with 5 example telecom feedbacks."""
    sample_data = pd.DataFrame({
        "feedback": [
            "The 5G speed is amazing in my city center.",
            "Network coverage is terrible inside my office building.",
            "Customer support resolved my billing issue quickly.",
            "Recharge plans are too expensive compared to competitors.",
            "App keeps crashing when I try to check data balance.",
        ]
    })
    return sample_data.to_csv(index=False).encode("utf-8")


# ─── Public Entry Point ───────────────────────────────────────────────────────

def render_page():
    """Public entry point called by app.py page router."""

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1 — Header
    # ═══════════════════════════════════════════════════════════════════════

    st.title("📊 Batch Feedback Analysis")
    st.markdown("Upload a CSV file to analyze multiple customer feedbacks at once.")

    st.info(
        "**CSV Requirements:**\n"
        "- Must contain a **\"feedback\"** column (also accepts: text, review, comment, message)\n"
        "- One feedback per row\n"
        "- Maximum **1000 rows**\n"
        "- Download the sample template below to see the expected format"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2 — Sample template download
    # ═══════════════════════════════════════════════════════════════════════

    st.download_button(
        label="📥 Download Sample CSV Template",
        data=generate_sample_csv(),
        file_name="sample_telecom_feedback.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # Check if we already have results in session state
    # ═══════════════════════════════════════════════════════════════════════

    if "batch_results" in st.session_state and st.session_state["batch_results"] is not None:
        _render_results_section()
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3 — File upload
    # ═══════════════════════════════════════════════════════════════════════

    uploaded_file = st.file_uploader(
        "Upload your feedback CSV",
        type=["csv"],
        key="batch_csv_upload",
    )

    if uploaded_file is None:
        st.caption("Upload a CSV file to begin analysis.")
        return

    # Read and validate
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"Failed to read CSV file: {str(e)}")
        logger.error(f"CSV read error: {e}")
        return

    validation = validate_csv(df)

    if not validation["valid"]:
        st.error(f"❌ {validation['error']}")
        return

    if validation["warning"]:
        st.warning(f"⚠️ {validation['warning']}")

    # Show preview
    st.markdown(f"**Preview** (first 5 rows of {validation['valid_row_count']} valid feedbacks):")
    st.dataframe(df.head(5), use_container_width=True)
    st.markdown(f"✅ **{validation['valid_row_count']} valid rows** ready for analysis.")

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4 — Analysis controls
    # ═══════════════════════════════════════════════════════════════════════

    time_estimate = estimate_batch_time(validation["valid_row_count"])
    st.caption(f"⏱ Estimated time: {time_estimate}")

    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
    )

    if not run_clicked:
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Section 5 — Progress tracking
    # ═══════════════════════════════════════════════════════════════════════

    try:
        # Load pipeline
        pipeline = load_pipeline()

        # Clean data
        cleaned_df = clean_csv_for_inference(df)
        feedbacks = cleaned_df["feedback"].tolist()

        # Progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run batch inference
        predictions = run_batch_inference(pipeline, feedbacks, progress_bar, status_text)

        # Prepare results
        results_df = prepare_results_dataframe(cleaned_df, predictions)
        summary_stats = get_batch_summary_stats(results_df)

        # Store in session state
        st.session_state["batch_results"] = {
            "results_df": results_df,
            "summary_stats": summary_stats,
        }

        st.success("✅ Analysis complete!")
        st.rerun()  # Rerun to show results section cleanly

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        logger.error(traceback.format_exc())
        st.error(
            "❌ Analysis failed due to an unexpected error. "
            "Please check that model files are present and try again."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Results Section (rendered after inference completes)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_results_section():
    """Render the full results dashboard after batch inference."""

    batch_data = st.session_state["batch_results"]
    results_df = batch_data["results_df"]
    summary_stats = batch_data["summary_stats"]

    # ═══════════════════════════════════════════════════════════════════════
    # Section 6 — Summary metrics
    # ═══════════════════════════════════════════════════════════════════════

    render_summary_metrics_row(summary_stats)
    st.markdown("")
    render_top_insights(summary_stats)
    render_error_summary(results_df)
    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # Section 7 — Dashboard charts
    # ═══════════════════════════════════════════════════════════════════════

    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Overview", "🎯 Aspect Analysis", "🔬 Advanced Insights"])

    with tab1:
        st.plotly_chart(sentiment_distribution_pie(results_df), use_container_width=True)
        st.plotly_chart(positive_negative_trend_bar(results_df), use_container_width=True)

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(aspect_frequency_bar(results_df), use_container_width=True)
        with col_right:
            st.plotly_chart(aspect_sentiment_heatmap(results_df), use_container_width=True)
        st.plotly_chart(top_negative_aspects_bar(results_df), use_container_width=True)

    with tab3:
        st.plotly_chart(confidence_distribution_histogram(results_df), use_container_width=True)
        st.plotly_chart(feedback_length_vs_aspect_count_scatter(results_df), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Section 8 — Results table
    # ═══════════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("📋 Detailed Predictions")

    st.dataframe(
        results_df,
        use_container_width=True,
        column_config={
            "feedback": st.column_config.TextColumn("Feedback", max_chars=100),
            "overall_sentiment": st.column_config.TextColumn("Sentiment"),
            "aspect_count": st.column_config.NumberColumn("# Aspects"),
            "inference_time_ms": st.column_config.NumberColumn("Time (ms)", format="%.1f"),
            "detected_aspects": st.column_config.TextColumn("Aspects", max_chars=80),
        },
        hide_index=True,
    )

    # Download button
    st.download_button(
        label="📥 Download Results CSV",
        data=results_to_download_csv(results_df),
        file_name="absa_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Section 9 — Re-run option
    # ═══════════════════════════════════════════════════════════════════════

    st.divider()
    if st.button("🔄 Analyze Another File", use_container_width=True):
        st.session_state["batch_results"] = None
        # Clear file uploader by removing its key from session state
        if "batch_csv_upload" in st.session_state:
            del st.session_state["batch_csv_upload"]
        st.rerun()
