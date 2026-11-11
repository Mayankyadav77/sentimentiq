"""
test_feature_extraction.py - Unit tests for TF-IDF Feature Extraction Pipeline
"""

import os
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from src.feature_extraction import ReviewFeatureExtractor


@pytest.fixture
def sample_train_corpus():
    return [
        "love dress fits well and looks amazing",
        "terrible quality fabric tore after one wash",
        "average item fits okay nothing special",
        "not happy with the purchase poor stitching"
    ]


@pytest.fixture
def fitted_extractor(sample_train_corpus):
    extractor = ReviewFeatureExtractor(ngram_range=(1, 2), max_features=100, min_df=1)
    extractor.fit(sample_train_corpus)
    return extractor


def test_fit_transform_shape_and_type(fitted_extractor, sample_train_corpus):
    """Verify output is a Scipy sparse matrix with expected dimensions."""
    matrix = fitted_extractor.transform(sample_train_corpus)
    assert issparse(matrix)
    assert matrix.shape[0] == len(sample_train_corpus)
    assert matrix.shape[1] > 0
    assert matrix.shape[1] <= 100


def test_ngram_range_contains_unigrams_and_bigrams(fitted_extractor):
    """Verify vocabulary contains both single words and two-word combinations."""
    feature_names = fitted_extractor.get_feature_names()
    unigrams = [w for w in feature_names if " " not in w]
    bigrams = [w for w in feature_names if " " in w]
    
    assert len(unigrams) > 0, "No unigrams found in vocabulary"
    assert len(bigrams) > 0, "No bigrams found in vocabulary"
    assert "love" in unigrams
    assert "love dress" in bigrams


def test_empty_and_whitespace_inputs(fitted_extractor):
    """Ensure transforming empty strings does not throw exceptions and yields zero rows."""
    empty_matrix = fitted_extractor.transform("")
    assert empty_matrix.shape[0] == 1
    assert empty_matrix.nnz == 0

    whitespace_matrix = fitted_extractor.transform(["   ", ""])
    assert whitespace_matrix.shape[0] == 2
    assert whitespace_matrix.nnz == 0


def test_out_of_vocabulary_handling(fitted_extractor):
    """Ensure completely unseen tokens in incoming reviews are handled gracefully."""
    oov_review = "supercalifragilisticexpialidocious quantum pseudorandom"
    oov_matrix = fitted_extractor.transform(oov_review)
    assert oov_matrix.shape[0] == 1
    assert oov_matrix.nnz == 0


def test_zero_data_leakage(sample_train_corpus):
    """
    CRITICAL TEST: Words exclusive to test data must NOT enter the vocabulary.
    """
    test_only_corpus = ["exquisite sapphire silk emerald velvet"]
    extractor = ReviewFeatureExtractor(ngram_range=(1, 2), min_df=1)
    extractor.fit(sample_train_corpus)

    vocab = extractor.get_feature_names()
    for word in ["sapphire", "emerald", "velvet"]:
        assert word not in vocab, f"Data leakage detected! '{word}' leaked into training vocabulary."


def test_save_and_load_persistence(fitted_extractor, tmp_path):
    """Verify Joblib serialization restores exact vocabulary and transformation."""
    save_path = str(tmp_path / "test_vectorizer.pkl")
    fitted_extractor.save(save_path)
    assert os.path.exists(save_path)

    loaded_extractor = ReviewFeatureExtractor.load(save_path)
    assert loaded_extractor.is_fitted is True

    test_doc = "love dress quality"
    original_vec = fitted_extractor.transform(test_doc).toarray()
    loaded_vec = loaded_extractor.transform(test_doc).toarray()
    
    np.testing.assert_array_almost_equal(original_vec, loaded_vec)
