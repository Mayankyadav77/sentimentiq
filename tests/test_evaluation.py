"""
test_evaluation.py - Unit tests for model artifacts, evaluation metrics, and error analysis
"""

import os
import json
import pytest
import numpy as np
import joblib

from src.evaluate import compute_evaluation_metrics, perform_error_analysis

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def test_model_artifacts_exist():
    """Verify that all required model and vectorizer artifacts were saved."""
    required_files = [
        "sentiment_model.pkl",
        "model_naive_bayes.pkl",
        "model_logistic_regression.pkl",
        "model_linear_svc.pkl",
        "tfidf_vectorizer.pkl",
        "model_metadata.json"
    ]
    for filename in required_files:
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Missing artifact: {filename}"
        assert os.path.getsize(path) > 0, f"Artifact is empty: {filename}"


def test_metadata_structure_and_completeness():
    """Verify that model_metadata.json contains complete benchmarks and genuine metrics."""
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["champion_model_name"] in ["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"]
    assert meta["classes"] == ["Negative", "Neutral", "Positive"]
    assert meta["training_samples"] > 10000
    assert meta["validation_samples"] > 2000
    assert meta["test_samples"] > 2000
    assert meta["vocabulary_size"] <= 10000

    # Ensure all 3 algorithms are present in the validation benchmark
    bench = meta["validation_benchmark"]
    assert "Multinomial Naive Bayes" in bench
    assert "Logistic Regression" in bench
    assert "Linear SVM" in bench

    for name, res in bench.items():
        assert 0.0 <= res["accuracy"] <= 1.0
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert "confusion_matrix" in res
        assert len(res["confusion_matrix"]) == 3

    # Check error analysis samples
    assert "error_analysis_samples" in meta
    assert len(meta["error_analysis_samples"]) > 0
    sample = meta["error_analysis_samples"][0]
    assert "review_text" in sample
    assert "actual_sentiment" in sample
    assert "predicted_sentiment" in sample
    assert sample["actual_sentiment"] != sample["predicted_sentiment"]


def test_compute_evaluation_metrics():
    """Test metric calculation logic with known deterministic labels."""
    y_true = np.array(["Positive", "Negative", "Neutral", "Positive"])
    y_pred = np.array(["Positive", "Negative", "Positive", "Positive"])  # 1 misclassification
    labels = ["Negative", "Neutral", "Positive"]

    metrics = compute_evaluation_metrics(y_true, y_pred, labels=labels)
    assert metrics["accuracy"] == 0.75
    assert metrics["total_samples"] == 4
    assert len(metrics["confusion_matrix"]) == 3
    assert "Positive" in metrics["per_class"]
    assert "Negative" in metrics["per_class"]
    assert "Neutral" in metrics["per_class"]


def test_loaded_models_can_predict():
    """Verify all saved models can load and make valid predictions."""
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    test_text = ["absolutely gorgeous dress fits perfectly", "terrible material tore immediately"]
    X = vec.transform(test_text)

    for m_file in ["model_naive_bayes.pkl", "model_logistic_regression.pkl", "model_linear_svc.pkl", "sentiment_model.pkl"]:
        model = joblib.load(os.path.join(MODELS_DIR, m_file))
        preds = model.predict(X)
        assert len(preds) == 2
        for p in preds:
            assert p in ["Negative", "Neutral", "Positive"]
