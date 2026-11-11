"""
train.py - Machine Learning Model Training, Tuning, and Selection Suite
------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module trains, tunes, evaluates, and compares three core ML algorithms:
1. Multinomial Naive Bayes
2. Logistic Regression (with class_weight='balanced')
3. Linear Support Vector Machine (LinearSVC, with class_weight='balanced')

Workflow:
- Ingestion of stratified training & validation splits.
- Text preprocessing (negation-preserving) applied separately to train/val/test.
- Hyperparameter tuning via Stratified 3-Fold Cross-Validation using sklearn
  Pipelines — TF-IDF is re-fitted inside each fold, eliminating within-train
  cross-validation leakage.
- Model evaluation on independent validation split (Accuracy, Macro/Weighted F1,
  Confusion Matrix).
- Selection of best performing model based on Validation Macro F1.
- Holdout test evaluation of the winning model (evaluated strictly once).
- Error analysis on misclassified reviews.
- Serialization of Pipeline models, standalone TF-IDF vectorizer, and metadata.

Data-Leakage Guarantees
-----------------------
1. Text preprocessing and TF-IDF fitting are performed exclusively on the
   15,847-sample training split.
2. Inside GridSearchCV the sklearn Pipeline re-fits the TF-IDF step on the
   training portion of each CV fold, so held-out CV fold documents never
   influence vocabulary or IDF statistics.
3. The 3,396-sample test set is evaluated exactly once after final model
   selection.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.preprocessing import TextPreprocessor
from src.feature_extraction import ReviewFeatureExtractor
from src.evaluate import compute_evaluation_metrics, perform_error_analysis, save_metadata

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("TrainPipeline")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads pre-partitioned train, validation, and test datasets."""
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Processed datasets missing. Run src/data_loader.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def preprocess_and_vectorize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Cleans text and extracts TF-IDF features.
    CRITICAL: Fits vectorizer ONLY on train data to prevent data leakage.
    """
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)

    logger.info("Applying NLP preprocessing to train, validation, and test sets...")
    X_train_clean = preprocessor.preprocess_series(train_df["full_review"])
    X_val_clean = preprocessor.preprocess_series(val_df["full_review"])
    X_test_clean = preprocessor.preprocess_series(test_df["full_review"])

    # Feature Extractor: fit strictly on train
    extractor = ReviewFeatureExtractor(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )
    X_train_vec = extractor.fit_transform(X_train_clean)
    X_val_vec = extractor.transform(X_val_clean)
    X_test_vec = extractor.transform(X_test_clean)

    # Save fitted vectorizer artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    extractor.save(vectorizer_path)

    y_train = train_df["sentiment"].values
    y_val = val_df["sentiment"].values
    y_test = test_df["sentiment"].values

    return (X_train_vec, y_train), (X_val_vec, y_val), (X_test_vec, y_test), extractor


def build_pipelines() -> Dict[str, Any]:
    """
    Returns a dictionary of sklearn Pipelines, each containing a TF-IDF vectorizer
    and a classifier. Wrapping TF-IDF inside each Pipeline ensures it is re-fitted
    within every cross-validation fold, eliminating any within-train CV leakage.
    The final, winning Pipeline is serialized after being refitted on the full training
    set — so the held-out val/test sets are never touched.
    """

    # Shared TF-IDF configuration (must match preprocess_and_vectorize params)
    tfidf_params = dict(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )

    pipelines = {
        "Multinomial Naive Bayes": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf",   MultinomialNB()),
            ]),
            "param_grid": {"clf__alpha": [0.05, 0.1, 0.5, 1.0, 2.0]},
            "supports_proba": True,
        },
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]),
            "param_grid": {"clf__C": [0.1, 0.5, 1.0, 3.0, 5.0]},
            "supports_proba": True,
        },
        "Linear SVM": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf",   LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)),
            ]),
            "param_grid": {"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]},
            "supports_proba": False,  # LinearSVC uses decision_function, not predict_proba
        },
    }
    return pipelines


def train_and_tune_models(X_train_text: pd.Series, y_train, cv_folds: int = 3) -> Dict[str, Any]:
    """
    Performs GridSearch cross-validation using sklearn Pipelines.

    TF-IDF is fitted INSIDE each CV fold (via the Pipeline), so vocabulary and
    IDF weights are never computed on held-out fold documents. This prevents
    within-train cross-validation leakage.

    Parameters
    ----------
    X_train_text : pd.Series
        Raw (preprocessed) text strings — NOT yet vectorized.
    y_train : array-like
        Sentiment class labels for the training split.
    cv_folds : int
        Number of Stratified K-Fold splits (default 3).

    Returns
    -------
    Dict mapping model name -> {'model', 'best_params', 'cv_score', 'supports_proba'}
        'model' is the fitted best Pipeline (TF-IDF + classifier).
    """
    from sklearn.pipeline import Pipeline
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    pipeline_specs = build_pipelines()
    trained_models = {}

    for name, spec in pipeline_specs.items():
        logger.info("Tuning %s via Pipeline GridSearchCV (TF-IDF re-fit per fold)...", name)
        grid = GridSearchCV(
            spec["pipeline"],
            spec["param_grid"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,  # Refit best estimator on full X_train_text after CV
        )
        grid.fit(X_train_text, y_train)
        logger.info("%s best params: %s (CV Macro F1: %.4f)", name, grid.best_params_, grid.best_score_)
        trained_models[name] = {
            "model": grid.best_estimator_,  # This is a fully fitted Pipeline
            "best_params": {k.replace("clf__", ""): v for k, v in grid.best_params_.items()},
            "cv_score": round(float(grid.best_score_), 4),
            "supports_proba": spec["supports_proba"],
        }

    return trained_models


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the entire end-to-end training and evaluation pipeline.

    Pipeline-based cross-validation note
    -------------------------------------
    `train_and_tune_models` wraps TF-IDF inside each sklearn Pipeline so the
    vectorizer is *re-fitted within every CV fold*, preventing within-train
    CV leakage.  After champion selection the pipeline is evaluated on the
    independent held-out val and test splits that were never touched during
    any fitting step.

    For inference-time artefacts the champion Pipeline's internal TF-IDF is
    also exported as a standalone tfidf_vectorizer.pkl so the existing
    `src/predict.py` loading logic (which loads vectorizer and classifier
    separately) continues to work without modification.
    """
    start_time = time.time()
    train_df, val_df, test_df = load_data_splits()

    # Text preprocessing (negation-preserving normalization)
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)
    logger.info("Applying NLP preprocessing to train, validation, and test sets...")
    X_train_clean = preprocessor.preprocess_series(train_df["full_review"])
    X_val_clean   = preprocessor.preprocess_series(val_df["full_review"])
    X_test_clean  = preprocessor.preprocess_series(test_df["full_review"])

    y_train = train_df["sentiment"].values
    y_val   = val_df["sentiment"].values
    y_test  = test_df["sentiment"].values

    # --- Pipeline-based GridSearchCV (TF-IDF re-fitted per fold) ---
    candidate_models = train_and_tune_models(X_train_clean, y_train)

    labels = ["Negative", "Neutral", "Positive"]
    validation_results = {}
    best_model_name = None
    best_val_macro_f1 = -1.0

    print("\n" + "=" * 78)
    print("VALIDATION SET MODEL COMPARISON BENCHMARK")
    print("=" * 78)
    print(f"{'Model Name':<28} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'CV Macro F1':<11}")
    print("-" * 78)

    for name, item in candidate_models.items():
        pipeline = item["model"]
        y_val_pred = pipeline.predict(X_val_clean)
        metrics = compute_evaluation_metrics(y_val, y_val_pred, labels=labels)
        metrics["best_params"] = item["best_params"]
        metrics["cv_score"] = item["cv_score"]
        validation_results[name] = metrics

        print(
            f"{name:<28} | {metrics['accuracy']:<9.4f} | {metrics['macro_f1']:<9.4f} | "
            f"{metrics['weighted_f1']:<11.4f} | {item['cv_score']:<11.4f}"
        )

        if metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = metrics["macro_f1"]
            best_model_name = name

    print("=" * 78)
    logger.info("Selected Best Model based on Validation Macro F1: %s (%.4f)", best_model_name, best_val_macro_f1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Serialize each Pipeline individually for multi-model UI switcher ---
    model_file_map = {
        "Multinomial Naive Bayes": "model_naive_bayes.pkl",
        "Logistic Regression":     "model_logistic_regression.pkl",
        "Linear SVM":              "model_linear_svc.pkl",
    }
    for name, filename in model_file_map.items():
        save_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(candidate_models[name]["model"], save_path)
        logger.info("Saved %s to %s", name, save_path)

    # Save champion Pipeline to sentiment_model.pkl
    best_pipeline = candidate_models[best_model_name]["model"]
    champion_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    joblib.dump(best_pipeline, champion_path)
    logger.info("Saved champion model (%s) to %s", best_model_name, champion_path)

    # --- Export standalone TF-IDF artefact for predict.py compatibility ---
    # The champion Pipeline's internal TF-IDF was fitted on the full training
    # set (GridSearchCV refits after CV); extract and save it separately.
    fitted_tfidf = best_pipeline.named_steps["tfidf"]
    extractor = ReviewFeatureExtractor(
        ngram_range=fitted_tfidf.ngram_range,
        max_features=fitted_tfidf.max_features,
        min_df=fitted_tfidf.min_df,
        max_df=fitted_tfidf.max_df,
        sublinear_tf=fitted_tfidf.sublinear_tf,
    )
    extractor.vectorizer = fitted_tfidf
    extractor.is_fitted = True
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    extractor.save(vectorizer_path)
    logger.info("Standalone TF-IDF vectorizer saved to %s (vocab: %d)", vectorizer_path, len(fitted_tfidf.vocabulary_))

    # --- Evaluate champion on the held-out test set (evaluated ONCE) ---
    logger.info("Evaluating champion model on the held-out test set...")
    y_test_pred = best_pipeline.predict(X_test_clean)
    test_metrics = compute_evaluation_metrics(y_test, y_test_pred, labels=labels)

    y_test_probs = None
    if hasattr(best_pipeline, "predict_proba"):
        y_test_probs = best_pipeline.predict_proba(X_test_clean)

    error_analysis_samples = perform_error_analysis(
        test_df, y_test, y_test_pred, y_probs=y_test_probs, max_samples=10
    )

    elapsed_time = round(time.time() - start_time, 2)

    metadata = {
        "platform": "SentimentIQ – Customer Review Intelligence Platform",
        "champion_model_name": best_model_name,
        "classes": labels,
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "vocabulary_size": int(len(fitted_tfidf.vocabulary_)),
        "elapsed_seconds": elapsed_time,
        "validation_benchmark": validation_results,
        "test_evaluation_champion": test_metrics,
        "error_analysis_samples": error_analysis_samples,
        "feature_parameters": {
            "ngram_range": [1, 2],
            "max_features": 10000,
            "min_df": 2,
            "max_df": 0.90,
            "sublinear_tf": True,
        },
        "cv_note": (
            "Cross-validation used sklearn Pipelines: TF-IDF was re-fitted within "
            "each fold so IDF statistics were never computed on held-out fold documents."
        ),
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    save_metadata(metadata, metadata_path)

    print("\n" + "=" * 78)
    print(f"CHAMPION MODEL TEST SET EVALUATION ({best_model_name})")
    print("=" * 78)
    print(f"Test Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"Test Macro Precision : {test_metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall    : {test_metrics['macro_recall']:.4f}")
    print(f"Test Macro F1-Score  : {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1     : {test_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Performance on Test Set:")
    for cls_name, cls_metrics in test_metrics["per_class"].items():
        print(
            f"  Class '{cls_name:<8}': Precision={cls_metrics['precision']:.4f}, "
            f"Recall={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f} "
            f"(Support={cls_metrics['support']})"
        )
    print("=" * 78)

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
