"""
Summary Metric Cards Component for Batch Dashboard.

Renders KPI metric cards, key insights, and error summaries
for the batch processing results page.
"""

import logging
import os
import sys

import pandas as pd

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from app.utils.app_helpers import format_aspect_name, get_sentiment_emoji

logger = logging.getLogger(__name__)


def render_summary_metrics_row(summary_stats: dict):
    """
    Render a row of 4 st.metric() cards showing key batch stats.

    Cards:
        1. Feedbacks Analyzed — total rows processed
        2. Aspects Detected — rows with at least one aspect
        3. Dominant Sentiment — most common overall_sentiment with emoji
        4. Avg Aspects/Feedback — rounded to 1 decimal

    Args:
        summary_stats: Dict from get_batch_summary_stats()
    """
    import streamlit as st

    total = summary_stats.get("total_rows", 0)
    with_aspects = summary_stats.get("rows_with_aspects", 0)
    avg_aspects = summary_stats.get("avg_aspects_per_feedback", 0.0)

    # Determine dominant sentiment
    sent_counts = summary_stats.get("sentiment_distribution", {}).get("counts", {})
    if sent_counts:
        dominant = max(sent_counts, key=sent_counts.get)
        dominant_count = sent_counts[dominant]
        emoji = get_sentiment_emoji(dominant)
        dominant_display = f"{emoji} {dominant.capitalize()} ({dominant_count})"
    else:
        dominant_display = "—"

    # Render 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📝 Feedbacks Analyzed",
            value=total,
        )

    with col2:
        st.metric(
            label="🎯 Aspects Detected",
            value=with_aspects,
            delta=f"{with_aspects * 100 // max(total, 1)}% of total" if total > 0 else None,
        )

    with col3:
        st.metric(
            label="💬 Dominant Sentiment",
            value=dominant_display,
        )

    with col4:
        st.metric(
            label="📊 Avg Aspects/Feedback",
            value=f"{avg_aspects:.1f}",
        )

    logger.info(f"Rendered summary metrics: total={total}, aspects={with_aspects}, "
                f"dominant={dominant_display}, avg={avg_aspects:.1f}")


def render_top_insights(summary_stats: dict):
    """
    Render 3 key insight boxes based on batch results.

    Insights:
        1. Most mentioned aspect (st.info)
        2. Most positive aspect (st.success)
        3. Most negative aspect (st.warning if ratio > 0.6)

    Args:
        summary_stats: Dict from get_batch_summary_stats()
    """
    import streamlit as st

    # Insight 1: Most mentioned aspect
    most_common = summary_stats.get("most_common_aspect")
    if most_common:
        name = format_aspect_name(most_common)
        st.info(f"📡 **{name}** was the most discussed aspect in this batch.")
    else:
        st.info("ℹ️ No aspects were detected in the uploaded feedback.")

    # Insight 2: Most positive aspect
    most_positive = summary_stats.get("most_common_positive_aspect")
    if most_positive:
        name = format_aspect_name(most_positive)
        st.success(f"😊 **{name}** received the most positive sentiment.")

    # Insight 3: Most negative aspect
    most_negative = summary_stats.get("most_common_negative_aspect")
    if most_negative:
        name = format_aspect_name(most_negative)
        # Check if negative ratio is high (use counts if available)
        sent_counts = summary_stats.get("sentiment_distribution", {}).get("counts", {})
        neg_count = sent_counts.get("negative", 0)
        total = summary_stats.get("total_rows", 1)
        neg_ratio = neg_count / max(total, 1)

        if neg_ratio > 0.6:
            st.warning(f"⚠️ **{name}** has a high negative sentiment ratio — needs attention!")
        else:
            st.warning(f"😞 **{name}** received the most negative sentiment.")

    logger.info(f"Rendered insights: common={most_common}, positive={most_positive}, "
                f"negative={most_negative}")


def render_error_summary(results_df: pd.DataFrame):
    """
    Show error summary for failed predictions in the batch.

    If any rows have overall_sentiment == "error", displays a warning
    with count and an expandable section showing failed feedback texts.

    Args:
        results_df: Results DataFrame from prepare_results_dataframe()
    """
    import streamlit as st

    error_rows = results_df[results_df["overall_sentiment"] == "error"]

    if error_rows.empty:
        return  # No errors — render nothing

    error_count = len(error_rows)
    total = len(results_df)

    st.warning(
        f"⚠️ **{error_count} of {total} feedbacks** failed during analysis. "
        f"These rows were skipped."
    )

    with st.expander(f"View {error_count} failed row(s)", expanded=False):
        for _, row in error_rows.iterrows():
            row_num = row.get("original_row_number", "?")
            feedback = row.get("feedback", "")[:100]
            st.markdown(f"- **Row {row_num}:** {feedback}{'...' if len(str(row.get('feedback',''))) > 100 else ''}")

    logger.warning(f"Rendered error summary: {error_count} failures out of {total} rows")
