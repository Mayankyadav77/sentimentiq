"""
feature_extraction.py - Production TF-IDF Feature Extraction Pipeline
----------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module implements a reusable TF-IDF feature extraction pipeline using
Scikit-learn's TfidfVectorizer. It translates clean natural language review
tokens into high-dimensional numerical feature spaces for linear and probabilistic
classifiers.

Key Architecture Features:
1. Strict train-only fitting to guarantee ZERO data leakage.
2. Unigram + Bigram ngram_range=(1, 2) to capture negation phrases (e.g., "not good").
3. Sublinear term-frequency scaling (sublinear_tf=True) replacing linear count
   with 1 + log(tf) to dampen the influence of overly repeated words.
4. Robust bounds: min_df to eliminate rare typos and max_df to filter corpus-wide noise.
5. Joblib serialization with metadata persistence.
"""

import os
import logging
from typing import List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("FeatureExtraction")

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_VECTORIZER_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.pkl")


class ReviewFeatureExtractor:
    """
    Production-grade TF-IDF feature extractor configured for sentiment intelligence.

    Parameters:
    -----------
    ngram_range : Tuple[int, int], default=(1, 2)
        Lower and upper boundary of the range of n-values for different n-grams.
        (1, 2) captures both individual words and 2-word collocations.
    max_features : int, default=10000
        Maximum vocabulary size ordered by term frequency across the training corpus.
    min_df : int or float, default=2
        Ignore terms that appear in fewer than min_df documents (filters out one-off typos).
    max_df : float, default=0.90
        Ignore terms that appear in more than 90% of documents (filters out non-informative corpus buzzwords).
    sublinear_tf : bool, default=True
        Applies sublinear scaling: tf = 1 + log(tf) when tf > 0.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        min_df: Union[int, float] = 2,
        max_df: float = 0.90,
        sublinear_tf: bool = True
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted: bool = False

    def _init_vectorizer(self) -> TfidfVectorizer:
        """Instantiates internal Scikit-learn TfidfVectorizer with validated parameters."""
        return TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",  # Matches 1+ char words (including single letters if needed)
            lowercase=False  # Text is already normalized by our preprocessing module
        )

    def fit(self, texts: Union[List[str], pd.Series]) -> "ReviewFeatureExtractor":
        """
        Fits vocabulary and inverse document frequency (IDF) weights ONLY on training text.
        Guarantees zero data leakage.
        """
        logger.info("Fitting TF-IDF vectorizer on %d training documents...", len(texts))
        self.vectorizer = self._init_vectorizer()
        
        # Clean nulls or empty series to strings
        safe_texts = pd.Series(texts).fillna("").astype(str)
        self.vectorizer.fit(safe_texts)
        self.is_fitted = True
        
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted successfully. Vocabulary size: %d terms (max_features=%d)", vocab_size, self.max_features)
        return self

    def transform(self, texts: Union[str, List[str], pd.Series]) -> csr_matrix:
        """
        Transforms incoming text into sparse TF-IDF feature matrix using learned training vocabulary.
        Safely handles empty strings, single reviews, and batch inputs.
        """
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("ReviewFeatureExtractor is not fitted. Call .fit() or .load() before .transform().")

        if isinstance(texts, str):
            texts = [texts]
        
        safe_texts = pd.Series(texts).fillna("").astype(str)
        # In case all input texts are blank whitespace
        safe_texts = safe_texts.replace(r"^\s*$", "", regex=True)

        return self.vectorizer.transform(safe_texts)

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """Fits vocabulary on training text and transforms it in a single pass."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        """Returns array of all vocabulary terms corresponding to feature columns."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Extractor is not fitted.")
        return self.vectorizer.get_feature_names_out()

    def get_top_features_for_review(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Helper for interpretability: Returns the top K terms in a single review
        with their associated TF-IDF weights.
        """
        matrix = self.transform(text)
        if matrix.nnz == 0:
            return []

        feature_names = self.get_feature_names()
        row = matrix.getrow(0)
        
        # Extract indices and data
        indices = row.indices
        data = row.data
        
        # Sort descending by TF-IDF score
        sorted_pairs = sorted(zip([feature_names[i] for i in indices], data), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_k]

    def save(self, filepath: str = DEFAULT_VECTORIZER_PATH) -> str:
        """Serializes the fitted vectorizer object to disk via Joblib."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Cannot save unfitted vectorizer.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info("Saved fitted TfidfVectorizer to: %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str = DEFAULT_VECTORIZER_PATH) -> "ReviewFeatureExtractor":
        """Loads a serialized vectorizer from disk and restores extractor state."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found at: {filepath}")

        loaded_vectorizer = joblib.load(filepath)
        instance = cls(
            ngram_range=loaded_vectorizer.ngram_range,
            max_features=loaded_vectorizer.max_features,
            min_df=loaded_vectorizer.min_df,
            max_df=loaded_vectorizer.max_df,
            sublinear_tf=loaded_vectorizer.sublinear_tf
        )
        instance.vectorizer = loaded_vectorizer
        instance.is_fitted = True
        logger.info("Loaded TfidfVectorizer from %s (Vocab size: %d)", filepath, len(loaded_vectorizer.vocabulary_))
        return instance


if __name__ == "__main__":
    # Self-test demonstration using genuine sample review sentences
    demo_reviews = [
        "absolutely love dress fit true size super flatter versatile",
        "not happy fit fabric felt cheap not look like photo",
        "okay not great not terrible quality neither bad nor impressive",
        "fabric damaged zipper broken terrible quality not buy"
    ]

    extractor = ReviewFeatureExtractor(ngram_range=(1, 2), max_features=50, min_df=1)
    tfidf_matrix = extractor.fit_transform(demo_reviews)

    print("=" * 70)
    print("TF-IDF FEATURE EXTRACTION DEMO")
    print("=" * 70)
    print(f"Matrix Shape (Documents x Features): {tfidf_matrix.shape}")
    print(f"Number of Non-Zero Entries (nnz)    : {tfidf_matrix.nnz}")
    print(f"Sparsity                            : {100.0 * (1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2f}%")
    print("\nSample Vocabulary with Unigrams and Bigrams:")
    features = extractor.get_feature_names()
    print(", ".join(features[:25]))

    print("\nReview-Level Feature Weights (Top 5 terms per review):")
    for idx, rev in enumerate(demo_reviews, 1):
        top_terms = extractor.get_top_features_for_review(rev, top_k=4)
        print(f"\nReview #{idx}: '{rev[:50]}...'")
        for term, weight in top_terms:
            print(f"   -> '{term:<18}' : {weight:.4f}")
    print("=" * 70)
