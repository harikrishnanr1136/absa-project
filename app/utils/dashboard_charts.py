"""
Plotly Chart Functions for Batch Processing Dashboard.

All functions return plotly.graph_objects.Figure with transparent backgrounds
for clean Streamlit rendering.
"""

import json
import logging
import os
import sys
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from app.utils.app_helpers import format_aspect_name

logger = logging.getLogger(__name__)

# ─── Common Layout Config ─────────────────────────────────────────────────────
TRANSPARENT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#F39C12",
    "mixed": "#95A5A6",
    "error": "#BDC3C7",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sentiment Distribution Pie
# ═══════════════════════════════════════════════════════════════════════════════

def sentiment_distribution_pie(results_df: pd.DataFrame) -> go.Figure:
    """
    Pie chart of overall_sentiment distribution.

    Colors: green=positive, red=negative, amber=neutral, grey=mixed/error.
    Shows percentage and count in labels.
    """
    counts = results_df["overall_sentiment"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [SENTIMENT_COLORS.get(s, "#BDC3C7") for s in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+percent+value",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hole=0.3,
    )])

    fig.update_layout(
        title="Overall Sentiment Distribution",
        **TRANSPARENT_BG,
        showlegend=True,
        margin=dict(t=50, b=20, l=20, r=20),
    )

    logger.info("Generated sentiment_distribution_pie chart")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Aspect Frequency Bar
# ═══════════════════════════════════════════════════════════════════════════════

def aspect_frequency_bar(results_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of top N most mentioned aspects.
    Parses comma-separated detected_aspects column.
    """
    all_aspects = Counter()
    for aspects_str in results_df["detected_aspects"]:
        if aspects_str and str(aspects_str).strip():
            aspects = [a.strip() for a in str(aspects_str).split(",") if a.strip()]
            all_aspects.update(aspects)

    if not all_aspects:
        fig = go.Figure()
        fig.add_annotation(text="No aspects detected", showarrow=False, font=dict(size=16))
        fig.update_layout(**TRANSPARENT_BG)
        return fig

    top_aspects = all_aspects.most_common(top_n)
    names = [format_aspect_name(a) for a, _ in reversed(top_aspects)]
    counts = [c for _, c in reversed(top_aspects)]

    fig = go.Figure(data=[go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color="#3498DB",
        text=counts,
        textposition="outside",
    )])

    fig.update_layout(
        title="Most Mentioned Aspects",
        xaxis_title="Count",
        **TRANSPARENT_BG,
        margin=dict(t=50, b=30, l=150, r=30),
    )

    logger.info(f"Generated aspect_frequency_bar chart (top {top_n})")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Positive vs Negative Trend Bar
# ═══════════════════════════════════════════════════════════════════════════════

def positive_negative_trend_bar(results_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: positive/negative/neutral counts per aspect.
    Only includes aspects with at least 3 mentions.
    """
    aspect_sentiments = Counter()  # {(aspect, sentiment): count}

    for _, row in results_df.iterrows():
        try:
            sentiments = json.loads(row["aspect_sentiments"]) if isinstance(row["aspect_sentiments"], str) else row["aspect_sentiments"]
            for aspect, sent in sentiments.items():
                aspect_sentiments[(aspect, sent)] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Get aspects with >= 3 mentions
    aspect_totals = Counter()
    for (aspect, _), count in aspect_sentiments.items():
        aspect_totals[aspect] += count

    valid_aspects = [a for a, c in aspect_totals.items() if c >= 3]
    valid_aspects.sort(key=lambda a: aspect_totals[a], reverse=True)

    if not valid_aspects:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", showarrow=False, font=dict(size=16))
        fig.update_layout(**TRANSPARENT_BG)
        return fig

    formatted_names = [format_aspect_name(a) for a in valid_aspects]

    pos_counts = [aspect_sentiments.get((a, "positive"), 0) for a in valid_aspects]
    neg_counts = [aspect_sentiments.get((a, "negative"), 0) for a in valid_aspects]
    neu_counts = [aspect_sentiments.get((a, "neutral"), 0) for a in valid_aspects]

    fig = go.Figure(data=[
        go.Bar(name="Positive", x=formatted_names, y=pos_counts, marker_color="#2ECC71"),
        go.Bar(name="Negative", x=formatted_names, y=neg_counts, marker_color="#E74C3C"),
        go.Bar(name="Neutral", x=formatted_names, y=neu_counts, marker_color="#F39C12"),
    ])

    fig.update_layout(
        title="Positive vs Negative by Aspect",
        barmode="group",
        xaxis_title="Aspect",
        yaxis_title="Count",
        **TRANSPARENT_BG,
        margin=dict(t=50, b=80, l=40, r=20),
        xaxis_tickangle=-45,
    )

    logger.info("Generated positive_negative_trend_bar chart")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Aspect-Sentiment Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def aspect_sentiment_heatmap(results_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap: aspects on Y axis, sentiments on X axis, count as values.
    Annotated with count in each cell.
    """
    aspect_sentiments = Counter()

    for _, row in results_df.iterrows():
        try:
            sentiments = json.loads(row["aspect_sentiments"]) if isinstance(row["aspect_sentiments"], str) else row["aspect_sentiments"]
            for aspect, sent in sentiments.items():
                aspect_sentiments[(aspect, sent)] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Get unique aspects and sentiments
    aspects = sorted(set(a for (a, _) in aspect_sentiments.keys()),
                     key=lambda a: sum(aspect_sentiments.get((a, s), 0) for s in ["positive", "negative", "neutral"]),
                     reverse=True)
    sentiment_labels = ["positive", "negative", "neutral"]

    if not aspects:
        fig = go.Figure()
        fig.add_annotation(text="No data for heatmap", showarrow=False, font=dict(size=16))
        fig.update_layout(**TRANSPARENT_BG)
        return fig

    # Build matrix
    z = []
    for aspect in aspects:
        row = [aspect_sentiments.get((aspect, s), 0) for s in sentiment_labels]
        z.append(row)

    formatted_aspects = [format_aspect_name(a) for a in aspects]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=sentiment_labels,
        y=formatted_aspects,
        colorscale="RdYlGn",
        text=z,
        texttemplate="%{text}",
        hovertemplate="Aspect: %{y}<br>Sentiment: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Aspect-Sentiment Heatmap",
        **TRANSPARENT_BG,
        margin=dict(t=50, b=30, l=150, r=30),
    )

    logger.info("Generated aspect_sentiment_heatmap chart")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Confidence Distribution Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def confidence_distribution_histogram(results_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of all confidence scores across all predictions.
    Adds vertical lines at 0.65 (medium) and 0.85 (high) thresholds.
    """
    all_scores = []

    for conf_str in results_df["confidence_scores"]:
        try:
            scores = json.loads(conf_str) if isinstance(conf_str, str) else conf_str
            if isinstance(scores, dict):
                all_scores.extend(scores.values())
        except (json.JSONDecodeError, TypeError):
            pass

    if not all_scores:
        fig = go.Figure()
        fig.add_annotation(text="No confidence data", showarrow=False, font=dict(size=16))
        fig.update_layout(**TRANSPARENT_BG)
        return fig

    fig = go.Figure(data=[go.Histogram(
        x=all_scores,
        nbinsx=20,
        marker_color="#9B59B6",
        opacity=0.8,
    )])

    # Threshold lines
    fig.add_vline(x=0.65, line_dash="dash", line_color="#F39C12",
                  annotation_text="Medium (0.65)", annotation_position="top")
    fig.add_vline(x=0.85, line_dash="dash", line_color="#2ECC71",
                  annotation_text="High (0.85)", annotation_position="top")

    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        **TRANSPARENT_BG,
        margin=dict(t=50, b=40, l=40, r=20),
    )

    logger.info(f"Generated confidence_distribution_histogram ({len(all_scores)} scores)")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Top Negative Aspects Bar
# ═══════════════════════════════════════════════════════════════════════════════

def top_negative_aspects_bar(results_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of aspects by negative sentiment ratio.
    Only includes aspects with >= 3 mentions.
    Adds horizontal line at 0.5 (majority negative threshold).
    """
    aspect_total = Counter()
    aspect_negative = Counter()

    for _, row in results_df.iterrows():
        try:
            sentiments = json.loads(row["aspect_sentiments"]) if isinstance(row["aspect_sentiments"], str) else row["aspect_sentiments"]
            for aspect, sent in sentiments.items():
                aspect_total[aspect] += 1
                if sent == "negative":
                    aspect_negative[aspect] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Filter and compute ratio
    valid = [(a, aspect_negative.get(a, 0) / t)
             for a, t in aspect_total.items() if t >= 3]
    valid.sort(key=lambda x: x[1], reverse=True)

    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", showarrow=False, font=dict(size=16))
        fig.update_layout(**TRANSPARENT_BG)
        return fig

    names = [format_aspect_name(a) for a, _ in valid]
    ratios = [r for _, r in valid]

    # Color gradient from amber to red
    colors = [f"rgb({int(230 + 25 * r)}, {int(150 - 100 * r)}, {int(60 - 20 * r)})" for r in ratios]

    fig = go.Figure(data=[go.Bar(
        x=names,
        y=ratios,
        marker_color=colors,
        text=[f"{r:.0%}" for r in ratios],
        textposition="outside",
    )])

    fig.add_hline(y=0.5, line_dash="dash", line_color="#E74C3C",
                  annotation_text="50% threshold")

    fig.update_layout(
        title="Aspects by Negative Sentiment Ratio",
        xaxis_title="Aspect",
        yaxis_title="Negative Ratio",
        yaxis_range=[0, 1.1],
        **TRANSPARENT_BG,
        margin=dict(t=50, b=80, l=40, r=20),
        xaxis_tickangle=-45,
    )

    logger.info("Generated top_negative_aspects_bar chart")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Feedback Length vs Aspect Count Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def feedback_length_vs_aspect_count_scatter(results_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: X = feedback word count, Y = aspect_count.
    Colored by overall_sentiment. Includes numpy polyfit trend line.
    """
    df = results_df.copy()
    df["word_count"] = df["feedback"].astype(str).apply(lambda x: len(x.split()))

    fig = go.Figure()

    # Plot each sentiment class separately for color legend
    for sent, color in SENTIMENT_COLORS.items():
        mask = df["overall_sentiment"] == sent
        subset = df[mask]
        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset["word_count"],
            y=subset["aspect_count"],
            mode="markers",
            name=sent.capitalize(),
            marker=dict(color=color, size=8, opacity=0.7),
            hovertemplate="Words: %{x}<br>Aspects: %{y}<br>Sentiment: " + sent + "<extra></extra>",
        ))

    # Trend line
    if len(df) > 2:
        x = df["word_count"].values
        y = df["aspect_count"].values
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                name="Trend",
                line=dict(color="grey", dash="dash", width=2),
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Feedback Length vs Aspect Count",
        xaxis_title="Word Count",
        yaxis_title="Aspects Detected",
        **TRANSPARENT_BG,
        margin=dict(t=50, b=40, l=40, r=20),
    )

    logger.info("Generated feedback_length_vs_aspect_count_scatter chart")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Generate all charts with mock data
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("dashboard_charts.py — GENERATING ALL CHARTS")
    print("=" * 60)

    # Mock results dataframe
    mock_data = pd.DataFrame([
        {"feedback": "Network coverage is excellent in my area.",
         "detected_aspects": "network_coverage",
         "aspect_sentiments": json.dumps({"network_coverage": "positive"}),
         "confidence_scores": json.dumps({"network_coverage": 0.92}),
         "overall_sentiment": "positive", "inference_time_ms": 120, "aspect_count": 1},
        {"feedback": "Terrible internet speed since last week.",
         "detected_aspects": "internet_speed",
         "aspect_sentiments": json.dumps({"internet_speed": "negative"}),
         "confidence_scores": json.dumps({"internet_speed": 0.88}),
         "overall_sentiment": "negative", "inference_time_ms": 115, "aspect_count": 1},
        {"feedback": "Customer support was very helpful today.",
         "detected_aspects": "customer_support",
         "aspect_sentiments": json.dumps({"customer_support": "positive"}),
         "confidence_scores": json.dumps({"customer_support": 0.85}),
         "overall_sentiment": "positive", "inference_time_ms": 98, "aspect_count": 1},
        {"feedback": "5G is fast but billing is confusing.",
         "detected_aspects": "5g_experience, billing",
         "aspect_sentiments": json.dumps({"5g_experience": "positive", "billing": "negative"}),
         "confidence_scores": json.dumps({"5g_experience": 0.90, "billing": 0.76}),
         "overall_sentiment": "mixed", "inference_time_ms": 145, "aspect_count": 2},
        {"feedback": "Recharge plans are too expensive now.",
         "detected_aspects": "recharge_plans, pricing",
         "aspect_sentiments": json.dumps({"recharge_plans": "negative", "pricing": "negative"}),
         "confidence_scores": json.dumps({"recharge_plans": 0.82, "pricing": 0.79}),
         "overall_sentiment": "negative", "inference_time_ms": 130, "aspect_count": 2},
        {"feedback": "The network and call quality is fine.",
         "detected_aspects": "network_coverage, call_quality",
         "aspect_sentiments": json.dumps({"network_coverage": "neutral", "call_quality": "neutral"}),
         "confidence_scores": json.dumps({"network_coverage": 0.60, "call_quality": 0.55}),
         "overall_sentiment": "neutral", "inference_time_ms": 110, "aspect_count": 2},
        {"feedback": "Internet speed great with the 5G plan.",
         "detected_aspects": "internet_speed, 5g_experience",
         "aspect_sentiments": json.dumps({"internet_speed": "positive", "5g_experience": "positive"}),
         "confidence_scores": json.dumps({"internet_speed": 0.91, "5g_experience": 0.87}),
         "overall_sentiment": "positive", "inference_time_ms": 125, "aspect_count": 2},
        {"feedback": "Billing issues again this month.",
         "detected_aspects": "billing",
         "aspect_sentiments": json.dumps({"billing": "negative"}),
         "confidence_scores": json.dumps({"billing": 0.84}),
         "overall_sentiment": "negative", "inference_time_ms": 105, "aspect_count": 1},
    ])

    # Output directory
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "dashboard_test")
    os.makedirs(out_dir, exist_ok=True)

    # Generate all charts
    charts = {
        "1_sentiment_pie": sentiment_distribution_pie(mock_data),
        "2_aspect_frequency": aspect_frequency_bar(mock_data),
        "3_pos_neg_trend": positive_negative_trend_bar(mock_data),
        "4_heatmap": aspect_sentiment_heatmap(mock_data),
        "5_confidence_hist": confidence_distribution_histogram(mock_data),
        "6_negative_ratio": top_negative_aspects_bar(mock_data),
        "7_length_scatter": feedback_length_vs_aspect_count_scatter(mock_data),
    }

    for name, fig in charts.items():
        path = os.path.join(out_dir, f"{name}.html")
        fig.write_html(path)
        print(f"  ✅ {name} → {path}")

    print(f"\n{'=' * 60}")
    print(f"ALL 7 CHARTS GENERATED → {out_dir}")
    print("=" * 60)
