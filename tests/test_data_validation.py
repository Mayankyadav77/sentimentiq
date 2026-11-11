"""
test_data_validation.py - Unit tests for data loading and schema validation
"""

import os
import pytest
import pandas as pd
from src.data_loader import (
    validate_schema,
    load_and_preprocess_dataset,
    RATING_TO_SENTIMENT,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH
)


def test_raw_dataset_exists():
    """Verify that the raw customer reviews dataset exists and is non-empty."""
    assert os.path.exists(RAW_DATA_PATH), f"Raw dataset missing at {RAW_DATA_PATH}"
    assert os.path.getsize(RAW_DATA_PATH) > 1000000, "Raw dataset file is unexpectedly small"


def test_schema_validation_success():
    """Test that schema validation passes when required columns exist."""
    valid_df = pd.DataFrame({
        "Review Text": ["Great fit and comfortable shirt!"],
        "Rating": [5]
    })
    assert validate_schema(valid_df) is True


def test_schema_validation_failure():
    """Test that schema validation raises ValueError when required columns are missing."""
    invalid_df = pd.DataFrame({
        "User_Comment": ["Nice product"],
        "Score": [4]
    })
    with pytest.raises(ValueError, match="Missing required column"):
        validate_schema(invalid_df)


def test_rating_to_sentiment_mapping():
    """Verify standard Likert scale mapping to 3-class sentiment."""
    assert RATING_TO_SENTIMENT[1] == "Negative"
    assert RATING_TO_SENTIMENT[2] == "Negative"
    assert RATING_TO_SENTIMENT[3] == "Neutral"
    assert RATING_TO_SENTIMENT[4] == "Positive"
    assert RATING_TO_SENTIMENT[5] == "Positive"


def test_processed_splits_exist_and_consistent():
    """Verify that train, validation, and test splits exist and have consistent schemas."""
    train_file = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_file = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_file = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    for path in [train_file, val_file, test_file]:
        assert os.path.exists(path), f"Split missing: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Split is empty: {path}"
        assert "full_review" in df.columns
        assert "sentiment" in df.columns
        assert "department_name" in df.columns
        # Verify sentiment labels contain all 3 classes
        assert set(df["sentiment"].unique()) == {"Positive", "Neutral", "Negative"}
