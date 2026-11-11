"""
test_prediction.py - Unit tests for Prediction Service and Explainability Engine
"""

import pytest
import numpy as np
import pandas as pd
from src.predict import SentimentPredictor, get_predictor


@pytest.fixture(scope="module")
def predictor():
    """Provides initialized SentimentPredictor singleton."""
    return get_predictor()


def test_positive_review_prediction(predictor):
    """Verify that clear positive reviews are classified as Positive with supporting words."""
    review = "Absolutely love this dress! Fits true to size, super flattering and amazing fabric."
    res = predictor.predict_single(review, model_name="Linear SVM")
    
    assert res["is_valid"] is True
    assert res["predicted_sentiment"] == "Positive"
    assert res["score_type"] == "decision_score"
    assert "Positive" in res["class_scores"]
    assert len(res["explanation"]["supporting_words"]) > 0
    words = [w["word"] for w in res["explanation"]["supporting_words"]]
    assert any(w in words for w in ["love", "flattering", "amazing", "great", "fits"])


def test_negative_review_with_negation(predictor):
    """Verify that negation phrases correctly lead to Negative sentiment."""
    review = "The product is not good. Poor stitching, fabric tore, will not buy again."
    res = predictor.predict_single(review, model_name="Linear SVM")
    
    assert res["is_valid"] is True
    assert res["predicted_sentiment"] == "Negative"
    assert len(res["explanation"]["supporting_words"]) > 0
    words = [w["word"] for w in res["explanation"]["supporting_words"]]
    # 'poor' or 'not' or 'not buy' should appear as drivers
    assert any(w in words for w in ["poor", "not", "not buy", "tore", "buy"])


def test_neutral_review_prediction(predictor):
    """Verify that moderate/mixed feedback is correctly handled."""
    review = "It fits okay. Nothing special, average quality for the price."
    res = predictor.predict_single(review, model_name="Linear SVM")
    
    assert res["is_valid"] is True
    assert res["predicted_sentiment"] in ["Neutral", "Negative"]  # Neutral can be close to negative
    assert "Neutral" in res["class_scores"]


def test_empty_and_whitespace_input(predictor):
    """Verify graceful handling of empty or whitespace reviews without raising exceptions."""
    for bad_input in ["", "   ", "\n\t", None]:
        res = predictor.predict_single(bad_input)
        assert res["is_valid"] is False
        assert res["confidence"] == 0.0
        assert "empty" in res["message"].lower() or "valid" in res["message"].lower()


def test_unseen_words_and_gibberish(predictor):
    """Verify handling of completely novel vocabulary."""
    gibberish = "qwertyuiop asdfghjkl zxcvbnm 123456789"
    res = predictor.predict_single(gibberish)
    assert "predicted_sentiment" in res
    assert res["predicted_sentiment"] in ["Negative", "Neutral", "Positive"]


def test_probability_support_for_probabilistic_models(predictor):
    """Ensure Logistic Regression and Naive Bayes return true normalized probabilities."""
    review = "Outstanding product, high quality and beautiful design."
    
    for model_name in ["Logistic Regression", "Multinomial Naive Bayes"]:
        res = predictor.predict_single(review, model_name=model_name)
        assert res["score_type"] == "probability"
        probs = list(res["class_scores"].values())
        # Check valid probability bounds
        for p in probs:
            assert 0.0 <= p <= 1.0
        assert sum(probs) == pytest.approx(1.0, abs=0.01)


def test_batch_prediction_dataframe(predictor):
    """Verify vectorized batch inference across multiple reviews."""
    batch_reviews = [
        "Love this item, five stars!",
        "Terrible item, ripped immediately.",
        "Fits okay, nothing to write home about.",
        ""
    ]
    df_out = predictor.predict_batch(batch_reviews, model_name="Linear SVM")
    
    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == 4
    assert "predicted_sentiment" in df_out.columns
    assert "cleaned_review" in df_out.columns
    assert df_out["predicted_sentiment"].iloc[0] == "Positive"
    assert df_out["predicted_sentiment"].iloc[1] == "Negative"
