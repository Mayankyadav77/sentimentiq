"""
preprocessing.py - Reusable Text Preprocessing Pipeline for Sentiment Analysis
-------------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module implements a production-grade NLP preprocessing pipeline that cleans,
normalizes, tokenizes, and filters customer review text while strictly preserving
negation semantics (e.g., 'not', 'never', 'no', 'without').

Key Pipeline Stages:
1. Input validation & null handling
2. HTML tag & URL removal
3. Contraction expansion (e.g., "wasn't" -> "was not")
4. Negation preservation during stopword filtering
5. Special character cleaning & whitespace normalization
6. Tokenization & optional lemmatization
"""

import re
import string
import logging
from typing import List, Optional, Set, Union
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logger
logger = logging.getLogger("Preprocessing")

# Contraction mapping to expand informal customer language and expose negations
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not",
    "it's": "it is",
    "i'm": "i am",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "there's": "there is",
    "that's": "that is"
}

# Negation words that MUST NOT be removed because they reverse polarity
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "nor", "neither", "cannot", "without",
    "barely", "hardly", "scarcely", "rarely", "seldom"
}

# Regex compilation for fast vector processing
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>|&[a-z]+;")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


class TextPreprocessor:
    """
    Modular, thread-safe text preprocessor for customer review sentiment intelligence.
    
    Attributes:
    -----------
    remove_stopwords : bool
        Whether to remove non-negation English stopwords.
    lemmatize : bool
        Whether to apply WordNet lemmatization.
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Load NLTK stopwords and filter out negation words
        try:
            base_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            base_stopwords = set(stopwords.words("english"))
            
        # Retain negations
        self.stop_words = base_stopwords - NEGATION_WORDS

        # Initialize lemmatizer if needed
        self.lemmatizer = None
        if self.lemmatize:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Test call to ensure wordnet is loaded
                self.lemmatizer.lemmatize("testing", pos="v")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                self.lemmatizer = WordNetLemmatizer()

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions to expose latent negation words."""
        for contraction, expansion in CONTRACTION_MAP.items():
            text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.IGNORECASE)
        # Handle trailing n't
        text = re.sub(r"(\w+)n't\b", r"\1 not", text, flags=re.IGNORECASE)
        return text

    def clean_text(self, text: Union[str, float, None]) -> str:
        """
        Cleans and normalizes text:
        - Replaces NaNs / Nones with empty string
        - Removes HTML tags and URLs
        - Expands contractions
        - Lowers case
        - Removes special symbols/digits while keeping letters and whitespace
        - Normalizes repeated whitespace
        """
        if text is None or pd.isna(text):
            return ""

        text = str(text)

        # 1. URL removal
        text = URL_PATTERN.sub(" ", text)

        # 2. HTML tag removal
        text = HTML_PATTERN.sub(" ", text)

        # 3. Contraction expansion (before lowercasing/stripping punctuation)
        text = self.expand_contractions(text)

        # 4. Lowercase conversion
        text = text.lower()

        # 5. Remove numbers and special characters (retain only letters and spaces)
        text = NON_ALPHA_PATTERN.sub(" ", text)

        # 6. Normalize multiple whitespaces
        text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

        return text

    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenizes text into words, removes filtered stopwords (preserving negations),
        and optionally applies lemmatization.
        """
        if not text:
            return []

        # Split on whitespace (safe and fast after regex cleanup)
        tokens = text.split()

        # Stopword filtering with negation preservation
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]
        else:
            tokens = [w for w in tokens if len(w) > 1]

        # Optional lemmatization
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(w, pos="v") for w in tokens]

        return tokens

    def preprocess(self, text: Union[str, float, None]) -> str:
        """
        Full end-to-end transformation returning clean space-separated text.
        Consistent across training, validation, testing, and production inference.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return " ".join(tokens)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        """
        Applies preprocessing across a Pandas Series efficiently.
        """
        return series.fillna("").astype(str).apply(self.preprocess)


# Module-level default preprocessor singleton for convenience
_DEFAULT_PREPROCESSOR = TextPreprocessor(remove_stopwords=True, lemmatize=False)


def preprocess_text(
    text: Union[str, float, None],
    remove_stopwords: bool = True,
    lemmatize: bool = False
) -> str:
    """
    Convenience function for single string preprocessing.
    """
    if remove_stopwords and not lemmatize:
        return _DEFAULT_PREPROCESSOR.preprocess(text)
    
    preprocessor = TextPreprocessor(remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    return preprocessor.preprocess(text)


if __name__ == "__main__":
    sample_reviews = [
        "I was NOT happy with the fit! The fabric felt cheap and didn't look like the photo: <br/>https://example.com/item",
        "Absolutely love this dress! Fits true to size, super flattering and versatile. 5/5 stars!!!",
        "It was okay, not great, not terrible. Quality is neither bad nor impressive.",
        "   ",
        None,
        "Won't buy again. Terrible customer service & damaged zipper!!!"
    ]

    print("=" * 70)
    print("NLP PREPROCESSING PIPELINE DEMO & VERIFICATION")
    print("=" * 70)
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    for idx, rev in enumerate(sample_reviews, 1):
        processed = preprocessor.preprocess(rev)
        print(f"\n[Sample {idx}]")
        print(f"RAW   : {rev}")
        print(f"CLEAN : {processed}")
    print("\n" + "=" * 70)
