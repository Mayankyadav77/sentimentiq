"""
app.py - SentimentIQ: Customer Review Intelligence Platform
------------------------------------------------------------
Main application entry point, executive overview, and system portal.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analytics import compute_kpis, compute_sentiment_distribution
from src.predict import get_predictor

# Page configuration
st.set_page_config(
    page_title="SentimentIQ – Review Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS CSS Theme
st.markdown("""
<style>
    /* Global styles */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .badge-pos {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neu {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neg {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .feature-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metadata():
    """Loads model metadata and evaluation benchmarks."""
    meta_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_corpus_data():
    """Loads the full corpus (train + validation + test) for consistent executive KPIs.

    All five KPI cards use the same 22,639-row base so that totals, counts,
    and percentages are internally consistent.
    """
    base = os.path.dirname(__file__)
    paths = {
        "train": os.path.join(base, "data", "processed", "train.csv"),
        "val":   os.path.join(base, "data", "processed", "val.csv"),
        "test":  os.path.join(base, "data", "processed", "test.csv"),
    }
    frames = [pd.read_csv(p) for p in paths.values() if os.path.exists(p)]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def main():
    metadata = load_metadata()
    train_df = load_corpus_data()

    # Sidebar
    with st.sidebar:
        st.markdown("### **SentimentIQ Platform**")
        st.caption("v1.0.0 · Production ML Pipeline")
        st.markdown("---")
        st.markdown("**Navigation Guide:**")
        st.markdown("""
        - **Home Overview**: System architecture & KPIs
        - **1. Sentiment Analysis**: Real-time review scoring
        - **2. Bulk Analysis**: CSV batch ingestion & export
        - **3. Analytics Dashboard**: Visual distributions & trends
        - **4. Model Performance**: Algorithmic benchmark & reports
        """)
        st.markdown("---")
        st.markdown("**Champion Model:**")
        champ_name = metadata.get("champion_model_name", "Linear SVM")
        st.success(f"**{champ_name}** (Balanced)")
        st.caption(f"Vocabulary: {metadata.get('vocabulary_size', 10000):,} n-grams")
        st.caption("Training Split: Zero Data Leakage")

    # Header
    st.markdown('<div class="main-title">SentimentIQ – Customer Review Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enterprise NLP & Machine Learning Engine for Customer Sentiment Analytics, Feature Attribution, and Granular Business Insights.</div>', unsafe_allow_html=True)

    # Executive KPI Metric Cards
    # NOTE: All KPI cards use the full corpus (train + val + test = 22,639 reviews)
    # so that totals, counts, and percentages are internally consistent.
    if len(train_df) > 0:
        kpis = compute_kpis(train_df)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Corpus Reviews</div>
                <div class="kpi-value">{kpis['total_reviews']:,}</div>
                <div class="kpi-sub">Full Dataset (Train + Val + Test)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Positive Sentiment</div>
                <div class="kpi-value" style="color: #16a34a;">{kpis['positive_pct']}%</div>
                <div class="kpi-sub">{kpis['positive_count']:,} Reviews (4–5 Stars)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Neutral Sentiment</div>
                <div class="kpi-value" style="color: #d97706;">{kpis['neutral_pct']}%</div>
                <div class="kpi-sub">{kpis['neutral_count']:,} Reviews (3 Stars)</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Negative Sentiment</div>
                <div class="kpi-value" style="color: #dc2626;">{kpis['negative_pct']}%</div>
                <div class="kpi-sub">{kpis['negative_count']:,} Reviews (1–2 Stars)</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            test_acc = metadata.get("test_evaluation_champion", {}).get("accuracy", 0.8324) * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Champion Accuracy</div>
                <div class="kpi-value" style="color: #2563eb;">{test_acc:.1f}%</div>
                <div class="kpi-sub">Held-Out Test Set ({metadata.get('test_samples', 3396):,} reviews)</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            f"📊 KPI source: full corpus ({kpis['total_reviews']:,} reviews = "
            f"{metadata.get('training_samples', 0):,} train + "
            f"{metadata.get('validation_samples', 0):,} val + "
            f"{metadata.get('test_samples', 0):,} test). "
            "Champion accuracy evaluated strictly on the independent held-out test split."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core System Pillars
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Platform Architecture & Capabilities")
        st.markdown("""
        SentimentIQ is built from the ground up as a **production-style, modular NLP platform** adhering to strict software engineering and machine learning best practices:
        """)

        st.markdown("""
        <div class="feature-box">
            <b>1. Deterministic NLP Preprocessing Pipeline</b><br>
            Extracts informal text signals, handles negation preserving semantics (e.g., <i>'not comfortable'</i>, <i>'never recommend'</i>), expands contractions, removes HTML/URLs, and applies sublinear TF-IDF unigram + bigram feature extraction.
        </div>
        <div class="feature-box">
            <b>2. Multi-Model Benchmark Suite</b><br>
            Compares <b>Multinomial Naive Bayes</b>, <b>Logistic Regression</b>, and <b>Linear Support Vector Machine (LinearSVC)</b>. Employs 3-Fold Stratified Cross-Validation, balanced class weighting to handle real-world class skew, and selects models via Macro F1 on validation splits.
        </div>
        <div class="feature-box">
            <b>3. Token-Level Explainability Engine</b><br>
            Deconstructs individual model weights and hyperplane distances to show exactly which positive and negative words drove a specific review's classification—providing full transparency without claiming false causation.
        </div>
        <div class="feature-box">
            <b>4. Executive Business Analytics</b><br>
            Cross-tabulates department performance across product categories (Tops, Dresses, Bottoms, Jackets, etc.) to immediately alert managers to product quality anomalies.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Sentiment Distribution")
        if len(train_df) > 0:
            dist_df = compute_sentiment_distribution(train_df)
            fig = px.pie(
                dist_df,
                values="count",
                names="sentiment",
                color="sentiment",
                color_discrete_map={
                    "Positive": "#10b981",
                    "Neutral": "#f59e0b",
                    "Negative": "#ef4444"
                },
                hole=0.55
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Class proportions computed on the full corpus (Train + Val + Test = 22,639 reviews). Class weighting (`class_weight='balanced'`) mitigates the naturally positive e-commerce distribution during model training.")

    st.markdown("---")

    # Quick interactive test section
    st.markdown("### Quick Sentiment Test (Live Inference)")
    st.markdown("Enter any sample customer review to test the live prediction service:")

    col_input, col_pred = st.columns([3, 2])
    with col_input:
        sample_input = st.text_area(
            "Review Text:",
            value="I was not happy with this purchase. The material felt cheap and tore on the first wear.",
            height=100
        )
        test_btn = st.button("Analyze Review Sentiment", type="primary")

    with col_pred:
        if test_btn and sample_input:
            predictor = get_predictor()
            res = predictor.predict_single(sample_input)
            
            st.markdown(f"**Predicted Sentiment:**")
            if res["predicted_sentiment"] == "Positive":
                st.markdown(f'<span class="badge-pos" style="font-size:1.1rem;">POSITIVE</span>', unsafe_allow_html=True)
            elif res["predicted_sentiment"] == "Negative":
                st.markdown(f'<span class="badge-neg" style="font-size:1.1rem;">NEGATIVE</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-neu" style="font-size:1.1rem;">NEUTRAL</span>', unsafe_allow_html=True)

            st.markdown(f"**Confidence / Margin:** `{res['confidence']}` ({res['score_type']})")
            if res["explanation"].get("supporting_words"):
                st.markdown(f"**Top Influencing Terms:**")
                chips = " ".join([f"`{w['word']} (+{w['weight']})`" for w in res["explanation"]["supporting_words"][:3]])
                st.markdown(chips)
                st.caption(res["explanation"]["summary_text"])
        else:
            st.info("Click 'Analyze Review Sentiment' to run inference using the cached Champion model.")

    st.markdown("---")
    st.caption("SentimentIQ · Designed for AI/ML Engineer, Data Scientist, and Software Engineering portfolios.")


if __name__ == "__main__":
    main()
