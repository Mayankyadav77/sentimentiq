"""
test_preprocessing.py - Unit tests for NLP Text Preprocessing Pipeline
"""

import pytest
import pandas as pd
from src.preprocessing import TextPreprocessor, preprocess_text, NEGATION_WORDS


@pytest.fixture
def preprocessor():
    """Default preprocessor fixture."""
    return TextPreprocessor(remove_stopwords=True, lemmatize=False)


@pytest.fixture
def lemmatizing_preprocessor():
    """Preprocessor fixture with lemmatization enabled."""
    return TextPreprocessor(remove_stopwords=True, lemmatize=True)


def test_lowercase_conversion(preprocessor):
    raw = "EXCELLENT QUALITY AND BEAUTIFUL COLOR!"
    cleaned = preprocessor.clean_text(raw)
    assert cleaned == "excellent quality and beautiful color"


def test_html_tag_removal(preprocessor):
    raw = "This is a <b>fabulous</b> dress.<br/><p>Highly recommended!</p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<b>" not in cleaned
    assert "<br/>" not in cleaned
    assert "<p>" not in cleaned
    assert "fabulous" in cleaned
    assert "highly recommended" in cleaned


def test_url_removal(preprocessor):
    raw = "Check my styling here: https://example.com/item?id=123 or www.shop.com for pictures."
    cleaned = preprocessor.clean_text(raw)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert ".com" not in cleaned
    assert "check my styling" in cleaned


def test_special_characters_and_punctuation(preprocessor):
    raw = "Top-tier zipper! #1 rating... 100% cotton (soft & cozy) @home."
    cleaned = preprocessor.clean_text(raw)
    # Digits and symbols should be stripped, only alpha and single spaces remain
    assert "#" not in cleaned
    assert "%" not in cleaned
    assert "@" not in cleaned
    assert "100" not in cleaned
    assert cleaned == "top tier zipper rating cotton soft cozy home"


def test_whitespace_normalization(preprocessor):
    raw = "   Too    much     whitespace    between   words.\n\nNext line.  "
    cleaned = preprocessor.clean_text(raw)
    assert "  " not in cleaned
    assert cleaned == "too much whitespace between words next line"


def test_contraction_expansion(preprocessor):
    raw = "I didn't like it because it won't stretch and isn't flattering."
    cleaned = preprocessor.clean_text(raw)
    assert "did not" in cleaned
    assert "will not" in cleaned
    assert "is not" in cleaned


def test_negation_words_preserved(preprocessor):
    """
    CRITICAL TEST: Ensure negation tokens are NOT removed by stopword filtering.
    """
    raw = "The dress was not comfortable, never fit right, and has no lining."
    processed = preprocessor.preprocess(raw)
    
    assert "not" in processed.split(), "'not' was incorrectly removed"
    assert "never" in processed.split(), "'never' was incorrectly removed"
    assert "no" in processed.split(), "'no' was incorrectly removed"


def test_optional_lemmatization(lemmatizing_preprocessor):
    raw = "The seams were tearing and dresses were running small."
    processed = lemmatizing_preprocessor.preprocess(raw)
    # 'tearing' should lemmatize to 'tear', 'running' to 'run'
    tokens = processed.split()
    assert "tear" in tokens or "tearing" in tokens
    assert "run" in tokens or "running" in tokens


def test_empty_and_null_inputs(preprocessor):
    """Ensure pipeline never throws unhandled exceptions on null, empty, or whitespace-only inputs."""
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess("   ") == ""
    assert preprocessor.preprocess(None) == ""
    assert preprocessor.preprocess(float("nan")) == ""


def test_preprocess_series():
    """Verify consistent vector transformation across a pandas Series."""
    s = pd.Series([
        "Loved the dress!",
        None,
        "Didn't like the color.",
        "   "
    ])
    preprocessor = TextPreprocessor(remove_stopwords=True)
    res = preprocessor.preprocess_series(s)
    
    assert len(res) == 4
    assert res.iloc[0] == "loved dress"
    assert res.iloc[1] == ""
    assert "not" in res.iloc[2]
    assert res.iloc[3] == ""
