"""
analytics.py - Business Analytics, Aggregation, and KPI Engine
--------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module calculates genuine descriptive statistics, sentiment distributions,
category cross-tabulations, vocabulary frequencies, and review length metrics
from raw, processed, or bulk-predicted datasets. No synthetic statistics.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter

from src.preprocessing import TextPreprocessor

logger = logging.getLogger("Analytics")


def compute_kpis(df: pd.DataFrame, sentiment_col: str = "sentiment") -> Dict[str, Any]:
    """
    Calculates executive KPI figures for dashboard summary cards.
    """
    total = len(df)
    if total == 0:
        return {
            "total_reviews": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "neutral_pct": 0.0,
            "negative_pct": 0.0,
            "avg_words": 0,
            "avg_chars": 0
        }

    counts = df[sentiment_col].value_counts().to_dict()
    pos = int(counts.get("Positive", 0))
    neu = int(counts.get("Neutral", 0))
    neg = int(counts.get("Negative", 0))

    # Detect text column
    text_col = "review_text" if "review_text" in df.columns else ("full_review" if "full_review" in df.columns else None)
    avg_words = 0
    avg_chars = 0
    if text_col and text_col in df.columns:
        s_text = df[text_col].dropna().astype(str)
        avg_chars = int(s_text.str.len().mean()) if len(s_text) > 0 else 0
        avg_words = int(s_text.str.split().str.len().mean()) if len(s_text) > 0 else 0

    return {
        "total_reviews": total,
        "positive_count": pos,
        "neutral_count": neu,
        "negative_count": neg,
        "positive_pct": round(100.0 * pos / total, 1),
        "neutral_pct": round(100.0 * neu / total, 1),
        "negative_pct": round(100.0 * neg / total, 1),
        "avg_words": avg_words,
        "avg_chars": avg_chars
    }


def compute_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = "sentiment") -> pd.DataFrame:
    """
    Returns structured DataFrame of sentiment counts and percentage shares.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["sentiment", "count", "percentage"])

    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    total = counts["count"].sum()
    counts["percentage"] = (100.0 * counts["count"] / total).round(2)
    return counts


def compute_category_sentiment(
    df: pd.DataFrame,
    category_col: str = "department_name",
    sentiment_col: str = "sentiment"
) -> pd.DataFrame:
    """
    Computes cross-tabulation of product categories against sentiment classes.
    """
    if category_col not in df.columns or sentiment_col not in df.columns or len(df) == 0:
        return pd.DataFrame()

    ct = pd.crosstab(
        df[category_col],
        df[sentiment_col],
        margins=False
    ).reset_index()

    # Ensure all three sentiment columns exist
    for c in ["Positive", "Neutral", "Negative"]:
        if c not in ct.columns:
            ct[c] = 0

    ct["Total"] = ct["Positive"] + ct["Neutral"] + ct["Negative"]
    ct["Positive_Share_%"] = (100.0 * ct["Positive"] / ct["Total"]).round(1)
    ct["Negative_Share_%"] = (100.0 * ct["Negative"] / ct["Total"]).round(1)
    ct = ct.sort_values(by="Total", ascending=False).reset_index(drop=True)
    return ct


def extract_top_ngrams(
    df: pd.DataFrame,
    sentiment_value: str,
    text_col: str = "review_text",
    n: int = 1,
    top_k: int = 15
) -> List[Tuple[str, int]]:
    """
    Extracts authentic top frequent words or bigrams among reviews of a specific sentiment class.
    Filters out generic stopwords while preserving informative sentiment adjectives and nouns.
    """
    if sentiment_value not in df["sentiment"].values or text_col not in df.columns:
        return []

    subset = df[df["sentiment"] == sentiment_value][text_col].dropna().astype(str)
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)

    token_counts = Counter()

    for rev in subset:
        clean = preprocessor.clean_text(rev)
        tokens = preprocessor.tokenize_and_filter(clean)
        
        # Filter out structural verbs, domain noise, and functional words for cleaner sentiment keyword bars
        filter_words = {
            "dress", "shirt", "top", "wear", "ordered", "bought", "look", "size",
            "would", "could", "also", "even", "much", "well", "back", "still", "really",
            "get", "got", "way", "made", "one", "fits", "fabric", "color", "like", "not"
        }
        filtered = [t for t in tokens if t not in filter_words and len(t) > 2]

        if n == 1:
            token_counts.update(filtered)
        elif n == 2 and len(filtered) >= 2:
            bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)]
            token_counts.update(bigrams)

    return token_counts.most_common(top_k)


def compute_review_length_distribution(
    df: pd.DataFrame,
    text_col: str = "review_text",
    sentiment_col: str = "sentiment"
) -> pd.DataFrame:
    """
    Calculates word count and character count per review, segmented by sentiment.
    """
    if text_col not in df.columns or sentiment_col not in df.columns:
        return pd.DataFrame()

    valid_df = df.dropna(subset=[text_col]).copy()
    valid_df["char_count"] = valid_df[text_col].astype(str).str.len()
    valid_df["word_count"] = valid_df[text_col].astype(str).str.split().str.len()

    return valid_df[[sentiment_col, "word_count", "char_count"]]


def compute_temporal_trends(
    df: pd.DataFrame,
    date_col: str = "review_date",
    sentiment_col: str = "sentiment"
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Computes time-series monthly sentiment volume if a valid date column exists.
    Returns (None, message) if date information is not present.
    """
    # Check for possible date columns (avoid matching substrings like 'time' in 'sentiment')
    date_keywords = {"date", "time", "timestamp", "review_date", "review_time", "reviewdate", "reviewtime"}
    possible_cols = [
        c for c in df.columns 
        if c.lower() in date_keywords or any(k in c.lower().replace("-", "_").split("_") for k in ["date", "time", "timestamp"])
    ]
    if not possible_cols and date_col not in df.columns:
        return None, "No valid timestamp or date column found in this dataset. Time-series trends require date metadata."

    actual_date_col = date_col if date_col in df.columns else (possible_cols[0] if possible_cols else None)
    if not actual_date_col or actual_date_col not in df.columns:
        return None, "No valid timestamp or date column found in this dataset. Time-series trends require date metadata."

    try:
        df_copy = df.copy()
        df_copy["parsed_date"] = pd.to_datetime(df_copy[actual_date_col], errors="coerce")
        df_copy = df_copy.dropna(subset=["parsed_date"])

        if len(df_copy) < 5:
            return None, "Insufficient valid date records to generate a temporal trend."

        df_copy["period"] = df_copy["parsed_date"].dt.to_period("M").astype(str)
        trend = pd.crosstab(df_copy["period"], df_copy[sentiment_col]).reset_index()
        trend = trend.sort_values(by="period")
        return trend, "Temporal trends calculated successfully."
    except Exception as e:
        return None, f"Could not parse dates from column '{actual_date_col}': {str(e)}"
