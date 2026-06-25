"""
Telecom ABSA Analyzer — Streamlit Application Entry Point.

Handles page configuration, sidebar navigation, session state, and page routing.
No business logic lives here — all inference is delegated to src/inference.py.
"""

import logging
import os
import sys
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom ABSA Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Initialization ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

if "batch_csv_upload" not in st.session_state:
    st.session_state["batch_csv_upload"] = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────

# App title and description
st.sidebar.markdown("## 📡 Telecom ABSA Analyzer")
st.sidebar.markdown("Aspect-Based Sentiment Analysis for Telecom Customer Feedback")
st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.radio(
    "Navigation",
    options=[
        "📝 Single Feedback Analysis",
        "📊 Batch CSV Processing",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Model info section
st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.markdown("""
| Property | Value |
|----------|-------|
| Model | DistilBERT fine-tuned |
| Dataset | 1,000 samples |
| Aspects | 15 categories |
| Sentiments | Positive, Negative, Neutral |
| Batch limit | 1,000 rows |
| Supports | CSV upload |
""")

st.sidebar.markdown("---")

# About section
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    "This app detects telecom-specific aspects mentioned in customer feedback "
    "and classifies the sentiment expressed toward each aspect."
)

with st.sidebar.expander("📋 All 15 Aspects"):
    st.markdown("""
    1. Network Coverage
    2. Internet Speed
    3. Call Quality
    4. Customer Support
    5. Billing
    6. Recharge Plans
    7. Data Balance
    8. Roaming
    9. SIM Activation
    10. Mobile App Experience
    11. OTT Bundle Services
    12. Pricing
    13. Value for Money
    14. Data Validity
    15. 5G Experience
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • DistilBERT • scikit-learn")

# ─── Page Routing ─────────────────────────────────────────────────────────────

try:
    if page == "📝 Single Feedback Analysis":
        from app.views.single_feedback import render_page
        render_page()

    elif page == "📊 Batch CSV Processing":
        from app.views.batch_processing import render_page
        render_page()

except Exception as e:
    logger.error(f"Page rendering failed: {e}")
    logger.error(traceback.format_exc())
    st.error(
        f"⚠️ Something went wrong while loading this page.\n\n"
        f"**Error:** {str(e)}\n\n"
        f"**Traceback:**\n```\n{traceback.format_exc()}\n```\n\n"
        f"Please try refreshing the page. If the problem persists, "
        f"check that all model files are present in `outputs/models/`."
    )
