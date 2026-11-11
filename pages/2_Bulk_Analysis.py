"""
pages/2_Bulk_Analysis.py - Bulk Review Sentiment Ingestion and Batch Analysis
----------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform
"""

import os
import sys
import io
import streamlit as st
import pandas as pd
import plotly.express as px

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_predictor
from src.analytics import compute_kpis, compute_sentiment_distribution

st.set_page_config(
    page_title="Bulk Review Analysis – SentimentIQ",
    page_icon="📁",
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
    .kpi-mini {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    /* Ensure wide result tables scroll horizontally on narrow screens */
    .stDataFrame > div {
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="page-title">Bulk Customer Review Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload customer review CSV spreadsheets for vectorized batch sentiment scoring, distribution analytics, and enriched data exports.</div>', unsafe_allow_html=True)

    predictor = get_predictor()

    # Sidebar settings
    with st.sidebar:
        st.markdown("### Batch Configuration")
        model_name = st.selectbox(
            "Select Inference Model:",
            options=["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"],
            index=0
        )
        st.caption("Batch predictions are processed using vectorized sparse matrix multiplications for maximum throughput.")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Customer Reviews CSV:",
            type=["csv"],
            help="Upload any CSV file containing customer review text."
        )

    with col_info:
        st.markdown("##### Quick Demo Sample:")
        st.markdown("No dataset on hand? Load the pre-packaged benchmark sample (50 reviews) with 1 click:")
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_bulk_reviews.csv")
        load_sample = st.button("▶ Load Sample CSV (50 Reviews)", use_container_width=True)

    df_to_analyze = None

    if uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded `{uploaded_file.name}` with {len(df_to_analyze)} rows.")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {str(e)}")
            return
    elif load_sample and os.path.exists(sample_path):
        df_to_analyze = pd.read_csv(sample_path)
        st.info("Loaded pre-packaged test dataset (50 customer reviews from holdout test split).")

    if df_to_analyze is not None and len(df_to_analyze) > 0:
        st.markdown("---")
        st.markdown("### 1. Column Selection & Validation")

        # Automatically detect best review text column
        text_col_candidates = [c for c in df_to_analyze.columns if any(k in c.lower() for k in ["review", "text", "comment", "feedback"])]
        default_col_idx = df_to_analyze.columns.get_loc(text_col_candidates[0]) if text_col_candidates else 0

        col_select = st.selectbox(
            "Select Review Text Column:",
            options=list(df_to_analyze.columns),
            index=default_col_idx
        )

        st.dataframe(df_to_analyze.head(3), use_container_width=True, hide_index=True)

        if st.button("Execute Batch Sentiment Analysis", type="primary"):
            progress_bar = st.progress(0, text="Initializing preprocessing and vectorization...")
            
            with st.spinner("Classifying sentiment across all rows..."):
                reviews_series = df_to_analyze[col_select]
                progress_bar.progress(30, text="Applying negation-preserving text normalization...")
                
                # Perform vectorized batch inference
                batch_res_df = predictor.predict_batch(reviews_series, model_name=model_name)
                progress_bar.progress(80, text="Generating probability and decision metrics...")

                # Merge predictions back into original dataframe
                df_to_analyze["predicted_sentiment"] = batch_res_df["predicted_sentiment"]
                if "confidence" in batch_res_df.columns:
                    df_to_analyze["confidence"] = batch_res_df["confidence"]

                progress_bar.progress(100, text="Batch inference complete!")

            st.session_state["bulk_analyzed_df"] = df_to_analyze

    # Display results if present in session state
    if "bulk_analyzed_df" in st.session_state:
        res_df = st.session_state["bulk_analyzed_df"]
        kpis = compute_kpis(res_df, sentiment_col="predicted_sentiment")

        st.markdown("---")
        st.markdown("### 2. Batch Analysis Results & Summary")

        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1:
            st.markdown(f"""
            <div class="kpi-mini">
                <div style="font-size:0.8rem;color:#64748b;font-weight:600;">TOTAL ANALYZED</div>
                <div style="font-size:1.6rem;font-weight:700;">{kpis['total_reviews']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol2:
            st.markdown(f"""
            <div class="kpi-mini">
                <div style="font-size:0.8rem;color:#16a34a;font-weight:600;">POSITIVE REVIEWS</div>
                <div style="font-size:1.6rem;font-weight:700;color:#16a34a;">{kpis['positive_count']} ({kpis['positive_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol3:
            st.markdown(f"""
            <div class="kpi-mini">
                <div style="font-size:0.8rem;color:#d97706;font-weight:600;">NEUTRAL REVIEWS</div>
                <div style="font-size:1.6rem;font-weight:700;color:#d97706;">{kpis['neutral_count']} ({kpis['neutral_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol4:
            st.markdown(f"""
            <div class="kpi-mini">
                <div style="font-size:0.8rem;color:#dc2626;font-weight:600;">NEGATIVE REVIEWS</div>
                <div style="font-size:1.6rem;font-weight:700;color:#dc2626;">{kpis['negative_count']} ({kpis['negative_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual breakdown
        c_chart, c_filter = st.columns([1, 2])
        with c_chart:
            dist_data = compute_sentiment_distribution(res_df, sentiment_col="predicted_sentiment")
            fig = px.pie(
                dist_data,
                values="count",
                names="sentiment",
                color="sentiment",
                color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                hole=0.45
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c_filter:
            st.markdown("##### Filter Dataset by Sentiment:")
            filter_choice = st.radio(
                "Filter View:",
                options=["All Reviews", "Positive Only", "Neutral Only", "Negative Only"],
                horizontal=True
            )

            filtered_df = res_df
            if filter_choice == "Positive Only":
                filtered_df = res_df[res_df["predicted_sentiment"] == "Positive"]
            elif filter_choice == "Neutral Only":
                filtered_df = res_df[res_df["predicted_sentiment"] == "Neutral"]
            elif filter_choice == "Negative Only":
                filtered_df = res_df[res_df["predicted_sentiment"] == "Negative"]

            st.dataframe(filtered_df, use_container_width=True, height=220, hide_index=True)

        # Download button
        st.markdown("<br>", unsafe_allow_html=True)
        csv_buffer = io.StringIO()
        res_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="Download Enriched Sentiment CSV",
            data=csv_bytes,
            file_name="sentiment_analyzed_reviews.csv",
            mime="text/csv",
            type="primary"
        )


if __name__ == "__main__":
    main()
