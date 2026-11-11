"""
pages/3_Analytics_Dashboard.py - Executive Customer Sentiment Analytics
------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

Data source transparency:
- When displaying Ground Truth statistics, data is the full corpus
  (train + val + test = 22,639 reviews) so all KPIs and charts are consistent.
- When displaying Batch Predictions, data comes from the uploaded CSV
  that was analyzed on the Bulk Analysis page.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import (
    compute_kpis,
    compute_sentiment_distribution,
    compute_category_sentiment,
    extract_top_ngrams,
    compute_review_length_distribution,
    compute_temporal_trends
)

st.set_page_config(
    page_title="Analytics Dashboard – SentimentIQ",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .card-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
    }
    .kpi-strip {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.04);
    }
    .kpi-strip-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-strip-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.3;
    }
    .kpi-strip-sub {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .source-badge {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border-radius: 9999px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_full_corpus():
    """Loads the complete corpus (train + val + test = 22,639 reviews).

    Using the full corpus ensures that KPI counts, percentages, and charts
    are all derived from the same denominator and are internally consistent.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [
        os.path.join(base, "data", "processed", "train.csv"),
        os.path.join(base, "data", "processed", "val.csv"),
        os.path.join(base, "data", "processed", "test.csv"),
    ]
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def render_kpi_strip(df: pd.DataFrame, sentiment_col: str, source_label: str):
    """Renders a 5-card KPI strip with consistent data source labeling."""
    if len(df) == 0:
        st.warning("No data available for KPI calculation.")
        return

    kpis = compute_kpis(df, sentiment_col=sentiment_col)

    # Check all required counts are present (handle batch data with no text col)
    avg_words = kpis.get("avg_words", 0)
    avg_chars = kpis.get("avg_chars", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-strip-label">Total Reviews</div>
            <div class="kpi-strip-value">{kpis['total_reviews']:,}</div>
            <div class="kpi-strip-sub">{source_label}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-strip-label">Positive</div>
            <div class="kpi-strip-value" style="color:#16a34a;">{kpis['positive_pct']}%</div>
            <div class="kpi-strip-sub">{kpis['positive_count']:,} reviews</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-strip-label">Neutral</div>
            <div class="kpi-strip-value" style="color:#d97706;">{kpis['neutral_pct']}%</div>
            <div class="kpi-strip-sub">{kpis['neutral_count']:,} reviews</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-strip-label">Negative</div>
            <div class="kpi-strip-value" style="color:#dc2626;">{kpis['negative_pct']}%</div>
            <div class="kpi-strip-sub">{kpis['negative_count']:,} reviews</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        if avg_words > 0:
            st.markdown(f"""
            <div class="kpi-strip">
                <div class="kpi-strip-label">Avg Review Length</div>
                <div class="kpi-strip-value">{avg_words}</div>
                <div class="kpi-strip-sub">words · {avg_chars} chars</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-strip">
                <div class="kpi-strip-label">Avg Review Length</div>
                <div class="kpi-strip-value" style="color:#94a3b8;">N/A</div>
                <div class="kpi-strip-sub">No text column in batch data</div>
            </div>""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="page-title">Executive Sentiment Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Explore categorical sentiment trends, vocabulary drivers, and customer review behaviour across departments.</div>', unsafe_allow_html=True)

    # Data Source Selection
    source_choice = st.radio(
        "Select Analytics Data Source:",
        options=["Ground Truth – Full Corpus (22,639 Reviews)", "Uploaded Batch Predictions (if available)"],
        horizontal=True
    )

    if source_choice.startswith("Uploaded") and "bulk_analyzed_df" in st.session_state:
        df = st.session_state["bulk_analyzed_df"]
        sentiment_col = "predicted_sentiment"
        source_label = f"Batch CSV ({len(df):,} rows)"
        st.markdown('<span class="source-badge">📁 Source: Uploaded Batch CSV</span>', unsafe_allow_html=True)
    else:
        df = load_full_corpus()
        sentiment_col = "sentiment"
        source_label = "Full Corpus (Train + Val + Test)"
        st.markdown('<span class="source-badge">📊 Source: Ground Truth — Full Corpus (Train + Val + Test)</span>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data available. Please verify that `data/processed/` contains train.csv, val.csv, and test.csv.")
        return

    # ── Section 0: KPI Strip ─────────────────────────────────────────────────
    render_kpi_strip(df, sentiment_col=sentiment_col, source_label=source_label)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Sentiment Distribution (Pie + Bar) ────────────────────────
    st.markdown("### 1. Overall Sentiment Distribution")
    c1, c2 = st.columns([1, 1])

    dist_df = compute_sentiment_distribution(df, sentiment_col=sentiment_col)

    with c1:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown("##### Sentiment Share (Donut Chart)")
        fig_donut = px.pie(
            dist_df,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
            hole=0.5
        )
        fig_donut.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        st.markdown("##### Sentiment Volume (Bar Chart)")
        fig_bar = px.bar(
            dist_df,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
            text_auto=True
        )
        fig_bar.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="Sentiment Class",
            yaxis_title="Review Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Department Cross-Tabulation ───────────────────────────────
    category_col = "department_name" if "department_name" in df.columns else None
    if category_col:
        st.markdown("---")
        st.markdown("### 2. Category-Wise Department Sentiment Breakdown")
        st.markdown("Compare customer sentiment performance across specific merchandise divisions:")

        cat_df = compute_category_sentiment(df, category_col=category_col, sentiment_col=sentiment_col)

        if len(cat_df) > 0:
            melted_cat = cat_df.melt(
                id_vars=[category_col, "Total", "Positive_Share_%", "Negative_Share_%"],
                value_vars=["Positive", "Neutral", "Negative"],
                var_name="Sentiment",
                value_name="Review Count"
            )

            fig_cat = px.bar(
                melted_cat,
                x=category_col,
                y="Review Count",
                color="Sentiment",
                barmode="group",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                text_auto=True
            )
            fig_cat.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=80),
                xaxis_title="Department",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            with st.expander("View Department Performance Cross-Tabulation Table"):
                # Display-only columns with renamed headers to prevent truncation
                display_cat = cat_df.rename(columns={
                    "department_name": "Department",
                    "Positive_Share_%": "Pos Share %",
                    "Negative_Share_%": "Neg Share %"
                })
                st.dataframe(display_cat, use_container_width=True, hide_index=True)
    else:
        st.info("Department column (`department_name`) not found in the current dataset. This section is only available for the full corpus.")

    # ── Section 3: N-Gram Word Frequency ────────────────────────────────────
    text_col = "full_review" if "full_review" in df.columns else ("review_text" if "review_text" in df.columns else None)
    if text_col:
        st.markdown("---")
        st.markdown("### 3. Frequent Sentiment Word Signals (N-Gram Analysis)")
        st.markdown("High-frequency terms extracted from actual customer text after stopword and domain filtering:")

        w_col1, w_col2 = st.columns(2)

        with w_col1:
            st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
            st.markdown("##### Top Positive Review Terms")
            top_pos = extract_top_ngrams(df, sentiment_value="Positive", text_col=text_col, n=1, top_k=10)
            if top_pos:
                pos_df = pd.DataFrame(top_pos, columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=True)
                fig_pos_words = px.bar(
                    pos_df, x="Frequency", y="Word", orientation="h",
                    color_discrete_sequence=["#10b981"], text_auto=True
                )
                fig_pos_words.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Frequency",
                    yaxis_title=None
                )
                st.plotly_chart(fig_pos_words, use_container_width=True)
            else:
                st.info("No positive reviews found in current dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with w_col2:
            st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
            st.markdown("##### Top Negative Review Terms")
            top_neg = extract_top_ngrams(df, sentiment_value="Negative", text_col=text_col, n=1, top_k=10)
            if top_neg:
                neg_df = pd.DataFrame(top_neg, columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=True)
                fig_neg_words = px.bar(
                    neg_df, x="Frequency", y="Word", orientation="h",
                    color_discrete_sequence=["#ef4444"], text_auto=True
                )
                fig_neg_words.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Frequency",
                    yaxis_title=None
                )
                st.plotly_chart(fig_neg_words, use_container_width=True)
            else:
                st.info("No negative reviews found in current dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No text column found in the current dataset. N-gram analysis requires a `review_text` or `full_review` column.")

    # ── Section 4: Review Length Distribution ───────────────────────────────
    if text_col:
        st.markdown("---")
        st.markdown("### 4. Review Length Distribution by Sentiment")
        st.markdown("Do dissatisfied customers write longer reviews than satisfied customers?")

        length_df = compute_review_length_distribution(df, text_col=text_col, sentiment_col=sentiment_col)

        if len(length_df) > 0:
            fig_box = px.box(
                length_df,
                x=sentiment_col,
                y="word_count",
                color=sentiment_col,
                color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                points=False
            )
            fig_box.update_layout(
                height=340,
                yaxis_title="Word Count per Review",
                xaxis_title="Sentiment Class",
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Could not compute review length distribution for the selected data source.")

    # ── Section 5: Temporal Trends ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5. Temporal Trends Over Time")
    trend_df, trend_msg = compute_temporal_trends(df)
    if trend_df is not None:
        fig_trend = px.line(
            trend_df,
            x="period",
            y=[c for c in ["Positive", "Neutral", "Negative"] if c in trend_df.columns],
            color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}
        )
        fig_trend.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Month",
            yaxis_title="Review Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(
            f"**Dataset Observation:** {trend_msg} "
            "The Women's E-Commerce Clothing Reviews dataset does not include purchase timestamps. "
            "Upload a CSV with a `date` or `timestamp` column on the Bulk Analysis page to enable time-series curves."
        )


if __name__ == "__main__":
    main()
