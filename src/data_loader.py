"""
data_loader.py - Dataset Ingestion, Validation, and Partitioning Module
-----------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform
Dataset: Women's E-Commerce Clothing Reviews (Real customer reviews)
Source: Nick Brooks (Kaggle), Public Domain (CC0)

Responsibilities:
1. Fetch and store the genuine raw CSV dataset if not present.
2. Validate required columns, schema, and non-empty rows.
3. Clean missing values, whitespace anomalies, and duplicate records.
4. Derive documented 3-class sentiment labels from standard 5-star ratings:
   - Rating 1.0 & 2.0 -> 'Negative'
   - Rating 3.0       -> 'Neutral'
   - Rating 4.0 & 5.0 -> 'Positive'
5. Perform stratified train/validation/test partitioning to prevent data leakage.
"""

import os
import sys
import urllib.request
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataLoader")

RAW_DATA_URL = "https://raw.githubusercontent.com/gracenzmy/experiment/main/Womens%20Clothing%20E-Commerce%20Reviews.csv"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "customer_reviews.csv")

# Standard rating to 3-class sentiment mapping
RATING_TO_SENTIMENT = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}


def download_raw_dataset(url: str = RAW_DATA_URL, target_path: str = RAW_DATA_PATH) -> str:
    """
    Downloads the genuine customer reviews dataset from the public repository
    if not already present locally.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        logger.info("Dataset already exists at: %s (%d bytes)", target_path, os.path.getsize(target_path))
        return target_path

    logger.info("Downloading customer review dataset from: %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    logger.info("Dataset successfully downloaded to: %s (%d bytes)", target_path, len(data))
    return target_path


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validates presence of core columns required for NLP sentiment classification.
    """
    # Check for original column names or normalized column names
    col_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    required = ["review_text", "rating"]
    
    for req in required:
        if req not in col_map:
            raise ValueError(f"Schema validation failed: Missing required column '{req}'. Found: {list(df.columns)}")
    return True


def load_and_preprocess_dataset(
    file_path: str = RAW_DATA_PATH,
    min_review_length: int = 5
) -> pd.DataFrame:
    """
    Loads raw CSV, normalizes column names, cleans missing values,
    derives sentiment labels, and preserves category metadata.

    Parameters:
    -----------
    file_path : str
        Path to customer_reviews.csv
    min_review_length : int
        Minimum characters required in review text to prevent noisy trivial strings.

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with columns:
        ['clothing_id', 'age', 'title', 'review_text', 'full_review',
         'rating', 'sentiment', 'recommended', 'positive_feedback_count',
         'division_name', 'department_name', 'class_name']
    """
    if not os.path.exists(file_path):
        download_raw_dataset(target_path=file_path)

    logger.info("Loading raw dataset from %s", file_path)
    df = pd.read_csv(file_path)
    initial_count = len(df)
    logger.info("Raw records loaded: %d rows, %d columns", initial_count, df.shape[1])

    # Drop unnamed index columns if present
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Standardize column naming
    rename_mapping = {
        "Clothing ID": "clothing_id",
        "Age": "age",
        "Title": "title",
        "Review Text": "review_text",
        "Rating": "rating",
        "Recommended IND": "recommended",
        "Positive Feedback Count": "positive_feedback_count",
        "Division Name": "division_name",
        "Department Name": "department_name",
        "Class Name": "class_name"
    }
    df = df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns})

    # Validate schema
    validate_schema(df)

    # Clean missing values in review text
    df = df.dropna(subset=["review_text"]).copy()
    after_na_count = len(df)
    logger.info("Dropped %d rows with missing review text", initial_count - after_na_count)

    # Deduplicate reviews based on review_text and clothing_id
    df = df.drop_duplicates(subset=["clothing_id", "review_text"]).copy()
    logger.info("Dropped %d duplicate reviews. Remaining: %d", after_na_count - len(df), len(df))

    # Strip and filter out excessively short reviews
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() >= min_review_length].copy()

    # Fill NA in categorical metadata with 'Unknown'
    for col in ["division_name", "department_name", "class_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill NA in title with empty string
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        # Create full review composite text (Title + Review Text)
        df["full_review"] = np.where(
            df["title"].str.len() > 0,
            df["title"] + ". " + df["review_text"],
            df["review_text"]
        )
    else:
        df["full_review"] = df["review_text"]

    # Map ratings (1 to 5) to 3-class sentiment
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["sentiment"] = df["rating"].map(RATING_TO_SENTIMENT)

    # Reset index
    df = df.reset_index(drop=True)
    logger.info("Dataset successfully cleaned: %d valid records.", len(df))
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset into Stratified Training, Validation, and Test sets.
    Stratification ensures identical class proportions across all splits, preventing bias.

    Split ratio: 70% Train, 15% Validation, 15% Test.
    """
    logger.info("Partitioning data with stratified split (Train: 70%, Val: 15%, Test: 15%)")
    
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"]
    )

    # Next split train and val from train_val_df
    # val_size relative to train_val: val_size / (1 - test_size) = 0.15 / 0.85 approx 0.1765
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df["sentiment"]
    )

    logger.info("Split complete: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    if save_splits:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
        val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
        test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Splits saved to %s", PROCESSED_DATA_DIR)

    return train_df, val_df, test_df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes statistical and distribution metrics for EDA and reporting.
    """
    sentiment_dist = df["sentiment"].value_counts().to_dict()
    sentiment_pct = (df["sentiment"].value_counts(normalize=True) * 100).round(2).to_dict()
    rating_dist = df["rating"].value_counts().sort_index().to_dict()
    
    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "rating_distribution": rating_dist,
        "sentiment_counts": sentiment_dist,
        "sentiment_percentages": sentiment_pct,
        "departments": df["department_name"].value_counts().to_dict() if "department_name" in df.columns else {},
        "avg_review_length_chars": int(df["full_review"].str.len().mean()),
        "avg_review_length_words": int(df["full_review"].str.split().str.len().mean()),
    }
    return summary


if __name__ == "__main__":
    download_raw_dataset()
    cleaned_df = load_and_preprocess_dataset()
    summary = get_dataset_summary(cleaned_df)
    
    print("\n" + "=" * 60)
    print("DATASET INGESTION & VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Valid Reviews : {summary['total_records']}")
    print(f"Columns             : {summary['columns']}")
    print("\nRating Distribution (1-5 Stars):")
    for r, cnt in summary["rating_distribution"].items():
        print(f"  {r} Star : {cnt:6d} reviews")
    print("\nDerived Sentiment Class Distribution:")
    for s, cnt in summary["sentiment_counts"].items():
        pct = summary["sentiment_percentages"][s]
        print(f"  {s:<10} : {cnt:6d} ({pct}%)")
    print("\nProduct Departments:")
    for dep, cnt in summary["departments"].items():
        print(f"  {dep:<15} : {cnt:6d} reviews")
    print(f"\nAverage Length : {summary['avg_review_length_chars']} characters (~{summary['avg_review_length_words']} words)")
    print("=" * 60)

    # Perform and persist stratified splits
    train_df, val_df, test_df = split_dataset(cleaned_df)
