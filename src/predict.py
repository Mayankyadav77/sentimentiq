"""
predict.py - Production Sentiment Prediction and Inference Engine
-----------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module delivers real-time and batch sentiment inference with:
- Dynamic model selection (Linear SVM, Logistic Regression, Naive Bayes).
- Consistent text preprocessing and TF-IDF transformation.
- Transparent reporting of genuine class probabilities or decision scores.
- Integrated token-level explainability.
- Robust handling of empty, whitespace, and malformed inputs.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor
from src.feature_extraction import ReviewFeatureExtractor
from src.explainability import SentimentExplainer

logger = logging.getLogger("Predict")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_REGISTRY = {
    "Linear SVM": os.path.join(MODELS_DIR, "model_linear_svc.pkl"),
    "Logistic Regression": os.path.join(MODELS_DIR, "model_logistic_regression.pkl"),
    "Multinomial Naive Bayes": os.path.join(MODELS_DIR, "model_naive_bayes.pkl")
}
DEFAULT_CHAMPION_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


class SentimentPredictor:
    """
    Thread-safe, high-performance sentiment predictor for customer review intelligence.
    Caches model artifacts in memory to prevent repetitive disk I/O.
    """

    def __init__(self, default_model_name: str = "Linear SVM"):
        self.default_model_name = default_model_name
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)
        
        # In-memory artifact caches
        self._cached_models: Dict[str, Any] = {}
        self._cached_vectorizer: Optional[ReviewFeatureExtractor] = None
        self._feature_names: Optional[np.ndarray] = None
        self._class_labels: List[str] = ["Negative", "Neutral", "Positive"]

        # Eager load default vectorizer and model
        self._get_vectorizer()
        self._get_model(default_model_name)

    def _get_vectorizer(self) -> ReviewFeatureExtractor:
        """Loads and caches the fitted TF-IDF feature extractor."""
        if self._cached_vectorizer is None:
            if not os.path.exists(VECTORIZER_PATH):
                raise FileNotFoundError(
                    f"TF-IDF vectorizer artifact missing at {VECTORIZER_PATH}. Run src/train.py first."
                )
            self._cached_vectorizer = ReviewFeatureExtractor.load(VECTORIZER_PATH)
            self._feature_names = self._cached_vectorizer.get_feature_names()
        return self._cached_vectorizer

    def _get_model(self, model_name: str) -> Any:
        """Retrieves or loads requested model from cache.

        Handles both:
        - Bare sklearn estimator (.pkl saved before the Pipeline refactor)
        - sklearn Pipeline containing a 'tfidf' step and a 'clf' step (new format)

        When a Pipeline is loaded the vectorization step is bypassed in
        predict_single/predict_batch because the Pipeline handles it internally.
        A `_is_pipeline` flag is set so callers know which code path to use.
        """
        if model_name in self._cached_models:
            return self._cached_models[model_name]

        model_path = MODEL_REGISTRY.get(model_name, DEFAULT_CHAMPION_PATH)
        if not os.path.exists(model_path):
            # Fall back to champion sentiment_model.pkl if specific file not found
            model_path = DEFAULT_CHAMPION_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train.py first.")

        model = joblib.load(model_path)
        self._cached_models[model_name] = model
        # Determine class labels from the estimator inside the pipeline or bare model
        estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(estimator, "classes_"):
            self._class_labels = list(estimator.classes_)
        return model

    def predict_single(
        self,
        review_text: Union[str, float, None],
        model_name: Optional[str] = None,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Generates sentiment prediction and explanation for a single customer review.

        Parameters:
        -----------
        review_text : str, float, or None
            Raw text input provided by the user.
        model_name : str, optional
            'Linear SVM', 'Logistic Regression', or 'Multinomial Naive Bayes'.
        explain : bool
            Whether to compute token-level feature attributions.

        Returns:
        --------
        Dict containing prediction, scores/probabilities, confidence, and explanation.
        """
        chosen_model_name = model_name or self.default_model_name
        model = self._get_model(chosen_model_name)
        extractor = self._get_vectorizer()

        # Input validation
        if review_text is None or pd.isna(review_text):
            raw_str = ""
        else:
            raw_str = str(review_text).strip()

        if len(raw_str) < 3:
            return {
                "input_text": raw_str,
                "cleaned_text": "",
                "predicted_sentiment": "Neutral",
                "model_name": chosen_model_name,
                "score_type": "none",
                "confidence": 0.0,
                "class_scores": {c: 0.0 for c in self._class_labels},
                "explanation": {
                    "summary_text": "Input was empty or insufficient for analysis.",
                    "supporting_words": [],
                    "opposing_words": []
                },
                "is_valid": False,
                "message": "Please enter a valid review containing at least 3 characters."
            }

        # Consistent NLP Preprocessing
        clean_text = self.preprocessor.preprocess(raw_str)
        if not clean_text:
            # Review contained only non-alphabetic symbols or stopwords
            return {
                "input_text": raw_str,
                "cleaned_text": "",
                "predicted_sentiment": "Neutral",
                "model_name": chosen_model_name,
                "score_type": "none",
                "confidence": 0.333,
                "class_scores": {c: round(1.0 / len(self._class_labels), 4) for c in self._class_labels},
                "explanation": {
                    "summary_text": "Text contained only punctuation, numbers, or removed stopwords.",
                    "supporting_words": [],
                    "opposing_words": []
                },
                "is_valid": False,
                "message": "Review contained only symbols or uninformative terms."
            }

        # Determine whether the loaded artifact is a bare estimator or a Pipeline
        is_pipeline = hasattr(model, "named_steps")

        if is_pipeline:
            # Pipeline handles vectorization internally
            X_vec = self._get_vectorizer().transform(clean_text)  # for explainability only
            X_input = clean_text  # Pipeline takes raw preprocessed text
            pred_class = model.predict([clean_text])[0]
        else:
            # Legacy bare estimator: vectorize manually
            extractor = self._get_vectorizer()
            X_vec = extractor.transform(clean_text)
            X_input = X_vec
            pred_class = model.predict(X_vec)[0]

        # Resolve the actual sklearn estimator for score computation
        estimator = model.named_steps["clf"] if is_pipeline else model

        # Handle out-of-vocabulary reviews (only relevant for bare-estimator path)
        if not is_pipeline and X_vec.nnz == 0:
            return {
                "input_text": raw_str,
                "cleaned_text": clean_text,
                "predicted_sentiment": pred_class,
                "model_name": chosen_model_name,
                "score_type": "prior",
                "confidence": 0.50,
                "class_scores": {c: 0.0 for c in self._class_labels},
                "explanation": {
                    "summary_text": "Text words are unseen in training vocabulary. Classified by model prior bias.",
                    "supporting_words": [],
                    "opposing_words": []
                },
                "is_valid": True,
                "message": "Vocabulary warning: Words were unseen in the training dataset."
            }

        # Compute genuine probabilities or decision scores using the internal estimator
        if hasattr(estimator, "predict_proba"):
            if is_pipeline:
                probs = model.predict_proba([clean_text])[0]
            else:
                probs = estimator.predict_proba(X_vec)[0]
            score_type = "probability"
            class_scores = {str(c): round(float(p), 4) for c, p in zip(self._class_labels, probs)}
            confidence = float(np.max(probs))
        elif hasattr(estimator, "decision_function"):
            if is_pipeline:
                decision = model.decision_function([clean_text])[0]
            else:
                decision = estimator.decision_function(X_vec)[0]
            score_type = "decision_score"
            if np.ndim(decision) == 0:
                # Binary SVM decision
                class_scores = {"Negative": round(-float(decision), 4), "Positive": round(float(decision), 4)}
                confidence = round(float(abs(decision)), 4)
            else:
                class_scores = {str(c): round(float(d), 4) for c, d in zip(self._class_labels, decision)}
                # Margin confidence: difference between highest score and second highest
                sorted_scores = sorted(decision, reverse=True)
                margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])
                confidence = round(max(0.0, margin), 4)
        else:
            score_type = "unknown"
            class_scores = {str(pred_class): 1.0}
            confidence = 1.0

        # Explainability (always uses sparse TF-IDF vector from standalone vectorizer)
        explanation = {}
        if explain:
            explainer = SentimentExplainer(estimator, self._feature_names, self._class_labels)
            explanation = explainer.explain_instance(X_vec, pred_class, top_k=6)

        return {
            "input_text": raw_str,
            "cleaned_text": clean_text,
            "predicted_sentiment": pred_class,
            "model_name": chosen_model_name,
            "score_type": score_type,
            "confidence": round(confidence, 4),
            "class_scores": class_scores,
            "explanation": explanation,
            "is_valid": True,
            "message": "Prediction generated successfully."
        }

    def predict_batch(
        self,
        reviews: Union[List[str], pd.Series],
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Efficient vectorized batch inference for bulk CSV review analysis.
        Handles both bare sklearn estimators and Pipeline objects.
        """
        chosen_model_name = model_name or self.default_model_name
        model = self._get_model(chosen_model_name)
        extractor = self._get_vectorizer()

        # Series conversion
        series = pd.Series(reviews).fillna("").astype(str)
        clean_series = self.preprocessor.preprocess_series(series)

        is_pipeline = hasattr(model, "named_steps")

        if is_pipeline:
            # Pipeline takes preprocessed text strings directly
            predictions = model.predict(clean_series)
            X_matrix = None  # Scores pulled from pipeline below
        else:
            # Bare estimator: vectorize manually
            X_matrix = extractor.transform(clean_series)
            predictions = model.predict(X_matrix)

        results = {
            "cleaned_review": clean_series,
            "predicted_sentiment": predictions
        }

        # Resolve estimator for scoring
        estimator = model.named_steps["clf"] if is_pipeline else model

        if hasattr(estimator, "predict_proba"):
            if is_pipeline:
                probs = model.predict_proba(clean_series)
            else:
                probs = estimator.predict_proba(X_matrix)
            results["confidence"] = np.round(np.max(probs, axis=1), 4)
            for idx, c in enumerate(self._class_labels):
                results[f"prob_{c.lower()}"] = np.round(probs[:, idx], 4)
        elif hasattr(estimator, "decision_function"):
            if is_pipeline:
                decision = model.decision_function(clean_series)
            else:
                decision = estimator.decision_function(X_matrix)
            if decision.ndim == 1:
                results["decision_score"] = np.round(decision, 4)
            else:
                for idx, c in enumerate(self._class_labels):
                    results[f"score_{c.lower()}"] = np.round(decision[:, idx], 4)

        return pd.DataFrame(results)


# Default singleton instance
_DEFAULT_PREDICTOR: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Returns singleton SentimentPredictor instance."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = SentimentPredictor()
    return _DEFAULT_PREDICTOR


if __name__ == "__main__":
    predictor = get_predictor()

    test_samples = [
        "Absolutely love this cardigan! Beautiful colors, great fit, and soft fabric. 10/10!",
        "The dress was not good. Poor stitching, fabric tore on first wear. Would not recommend.",
        "It fits okay. Nothing special, average quality for the price.",
        "   ",
        "Zyzzxxvv completely alien words out of dictionary"
    ]

    print("=" * 78)
    print("SENTIMENT PREDICTION & EXPLAINABILITY ENGINE DEMO")
    print("=" * 78)

    for idx, s in enumerate(test_samples, 1):
        res = predictor.predict_single(s, model_name="Linear SVM")
        print(f"\n[Test #{idx}]")
        print(f"Input Review : {res['input_text'][:70]}")
        print(f"Cleaned Text : {res['cleaned_text'][:70]}")
        print(f"Prediction   : {res['predicted_sentiment']} (Confidence: {res['confidence']})")
        print(f"Scores       : {res['class_scores']}")
        if res["is_valid"] and res["explanation"].get("supporting_words"):
            top_w = [f"{w['word']} (+{w['weight']})" for w in res["explanation"]["supporting_words"][:3]]
            print(f"Top Drivers  : {', '.join(top_w)}")
            print(f"Explanation  : {res['explanation']['summary_text']}")
    print("=" * 78)
