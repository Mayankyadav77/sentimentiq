"""
pages/1_Sentiment_Analysis.py - Interactive Single Review Sentiment Analyzer
----------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_predictor

st.set_page_config(
    page_title="Single Review Analysis – SentimentIQ",
    page_icon="🔍",
    layout="wide"
)

# Custom styles
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
    .badge-pos-lg {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }
    .badge-neu-lg {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }
    .badge-neg-lg {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="page-title">Single Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Analyze individual customer reviews with real-time inference, model switching, and token-level explainability.</div>', unsafe_allow_html=True)

    predictor = get_predictor()

    # Model and settings sidebar
    with st.sidebar:
        st.markdown("### Model Configuration")
        model_choice = st.selectbox(
            "Select Inference Model:",
            options=["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"],
            index=0,
            help="Linear SVM is the champion model based on validation Macro F1."
        )

        st.markdown("---")
        st.markdown("**Model Characteristics:**")
        if model_choice == "Linear SVM":
            st.info("**Linear SVM (Champion):** Maximizes hyperplane margin. Uses `decision_function` margin scores.")
        elif model_choice == "Logistic Regression":
            st.info("**Logistic Regression:** Calibrated probability outputs via softmax with balanced class weighting.")
        else:
            st.info("**Multinomial Naive Bayes:** Probabilistic classifier modeling token frequency distributions.")

    # Preset Review buttons for convenient one-click testing
    st.markdown("##### Quick Test Presets:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_text = None
    if col_p1.button("Positive Praise", use_container_width=True):
        preset_text = "Absolutely love this dress! Fits true to size, super flattering and versatile. 5 stars!"
    if col_p2.button("Negative Complaint", use_container_width=True):
        preset_text = "Horrible quality! Fabric tore on the first wear and customer service was unhelpful."
    if col_p3.button("Neutral / Average", use_container_width=True):
        preset_text = "It was okay, not great, not terrible. Average quality for the price."
    if col_p4.button("Complex Negation", use_container_width=True):
        preset_text = "I really wanted to like it, but the dress was not comfortable and never fit properly."

    # Review text input
    default_review = preset_text or "The fabric was soft and comfortable, but the sizing was not true to size."
    review_input = st.text_area(
        "Enter Customer Review:",
        value=default_review,
        height=130,
        placeholder="Type or paste customer review text here..."
    )

    analyze_col, clear_col = st.columns([1, 5])
    with analyze_col:
        run_analysis = st.button("Classify Sentiment", type="primary", use_container_width=True)

    if review_input and (run_analysis or preset_text):
        with st.spinner("Analyzing review semantics and calculating feature weights..."):
            res = predictor.predict_single(review_input, model_name=model_choice, explain=True)

        if not res["is_valid"]:
            st.warning(f"{res['message']}")
            return

        st.markdown("<br>", unsafe_allow_html=True)

        # Output Layout: 2 Columns
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("#### Prediction Result")

            sentiment = res["predicted_sentiment"]
            if sentiment == "Positive":
                badge_html = f'<span class="badge-pos-lg">POSITIVE SENTIMENT</span>'
            elif sentiment == "Negative":
                badge_html = f'<span class="badge-neg-lg">NEGATIVE SENTIMENT</span>'
            else:
                badge_html = f'<span class="badge-neu-lg">NEUTRAL SENTIMENT</span>'

            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Score / Confidence Display
            if res["score_type"] == "probability":
                st.markdown(f"**Confidence (Softmax Probability):** `{res['confidence'] * 100:.1f}%`")
                # Class probability bar chart
                prob_df = pd.DataFrame([
                    {"Sentiment": k, "Probability (%)": v * 100}
                    for k, v in res["class_scores"].items()
                ])
                fig_prob = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Sentiment",
                    orientation="h",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                    text_auto=".1f"
                )
                fig_prob.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.markdown(f"**Decision Margin Distance:** `+{res['confidence']:.4f}`")
                st.caption("Linear SVM margin distance measures separating hyperplane distance. Positive values indicate separation magnitude.")
                
                score_df = pd.DataFrame([
                    {"Sentiment": k, "Decision Score": v}
                    for k, v in res["class_scores"].items()
                ])
                fig_score = px.bar(
                    score_df,
                    x="Decision Score",
                    y="Sentiment",
                    orientation="h",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                    text_auto=".2f"
                )
                fig_score.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_score, use_container_width=True)

            st.markdown(f"**Cleaned Tokens:** `{res['cleaned_text']}`")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("#### Token-Level Feature Attribution")

            explanation = res["explanation"]
            supporting = explanation.get("supporting_words", [])
            opposing = explanation.get("opposing_words", [])

            if supporting or opposing:
                records = []
                for item in supporting:
                    records.append({"Word": item["word"], "Weight": item["weight"], "Type": "Supporting (+)"})
                for item in opposing:
                    records.append({"Word": item["word"], "Weight": item["weight"], "Type": "Opposing (-)"})

                contrib_df = pd.DataFrame(records)
                contrib_df = contrib_df.sort_values(by="Weight", ascending=True)

                fig_contrib = px.bar(
                    contrib_df,
                    x="Weight",
                    y="Word",
                    orientation="h",
                    color="Type",
                    color_discrete_map={"Supporting (+)": "#10b981", "Opposing (-)": "#ef4444"},
                    text_auto=".3f"
                )
                fig_contrib.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

                st.markdown(f"**Explanation:** {explanation.get('summary_text', '')}")
            else:
                st.info("No specific vocabulary tokens had non-zero attribution scores.")

            st.caption(explanation.get("disclaimer", ""))
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
