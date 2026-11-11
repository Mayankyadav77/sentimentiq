"""
test_analytics.py - Unit tests for Business Analytics and KPI Engine
"""

import pytest
import pandas as pd
from src.analytics import (
    compute_kpis,
    compute_sentiment_distribution,
    compute_category_sentiment,
    extract_top_ngrams,
    compute_review_length_distribution,
    compute_temporal_trends
)


@pytest.fixture
def mock_dataset():
    """Generates synthetic dataframe for deterministic analytics testing."""
    return pd.DataFrame({
        "review_text": [
            "Love this beautiful dress fits great",
            "Terrible fabric tore immediately poor quality",
            "Average shirt fits okay nothing special",
            "Fantastic quality very comfortable",
            "Horrible zipper broke first day"
        ],
        "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"],
        "department_name": ["Dresses", "Tops", "Tops", "Dresses", "Bottoms"]
    })


def test_compute_kpis(mock_dataset):
    kpis = compute_kpis(mock_dataset)
    assert kpis["total_reviews"] == 5
    assert kpis["positive_count"] == 2
    assert kpis["negative_count"] == 2
    assert kpis["neutral_count"] == 1
    assert kpis["positive_pct"] == 40.0
    assert kpis["negative_pct"] == 40.0
    assert kpis["neutral_pct"] == 20.0
    assert kpis["avg_words"] > 0


def test_compute_sentiment_distribution(mock_dataset):
    dist = compute_sentiment_distribution(mock_dataset)
    assert len(dist) == 3
    assert "sentiment" in dist.columns
    assert "count" in dist.columns
    assert "percentage" in dist.columns
    assert dist["count"].sum() == 5


def test_compute_category_sentiment(mock_dataset):
    cat_df = compute_category_sentiment(mock_dataset, category_col="department_name")
    assert "department_name" in cat_df.columns
    assert "Positive" in cat_df.columns
    assert "Negative" in cat_df.columns
    assert "Neutral" in cat_df.columns
    assert "Total" in cat_df.columns
    assert len(cat_df) == 3  # Dresses, Tops, Bottoms


def test_extract_top_ngrams(mock_dataset):
    top_pos = extract_top_ngrams(mock_dataset, sentiment_value="Positive", text_col="review_text", top_k=5)
    assert len(top_pos) > 0
    words = [w[0] for w in top_pos]
    assert any(w in words for w in ["love", "beautiful", "great", "comfortable"])


def test_compute_review_length_distribution(mock_dataset):
    length_df = compute_review_length_distribution(mock_dataset, text_col="review_text")
    assert "word_count" in length_df.columns
    assert "char_count" in length_df.columns
    assert len(length_df) == 5


def test_compute_temporal_trends_missing_date(mock_dataset):
    trend_df, msg = compute_temporal_trends(mock_dataset)
    assert trend_df is None
    assert "no valid timestamp" in msg.lower()


def test_compute_temporal_trends_valid_date():
    df_with_date = pd.DataFrame({
        "review_text": ["Nice item", "Bad item", "Okay item", "Great item", "Poor item"],
        "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"],
        "review_date": ["2023-01-15", "2023-01-20", "2023-02-10", "2023-02-18", "2023-03-01"]
    })
    trend_df, msg = compute_temporal_trends(df_with_date)
    assert trend_df is not None
    assert "period" in trend_df.columns
    assert len(trend_df) == 3  # 3 months: 2023-01, 2023-02, 2023-03
