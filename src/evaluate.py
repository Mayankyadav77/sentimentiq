"""
evaluate.py - Model Evaluation, Metrics Generation, and Error Analysis Module
-----------------------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module calculates evaluation metrics, confusion matrices, classification
reports, and performs qualitative error analysis to inspect misclassifications.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger("Evaluate")


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics across all sentiment classes:
    Accuracy, Macro/Weighted Precision, Recall, and F1-Scores, plus per-class metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true)))

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # Per-class metrics
    class_report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    per_class = {}
    for lbl in labels:
        if lbl in class_report_dict:
            per_class[lbl] = {
                "precision": round(class_report_dict[lbl]["precision"], 4),
                "recall": round(class_report_dict[lbl]["recall"], 4),
                "f1": round(class_report_dict[lbl]["f1-score"], 4),
                "support": int(class_report_dict[lbl]["support"])
            }

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_precision": round(prec_weighted, 4),
        "weighted_recall": round(rec_weighted, 4),
        "weighted_f1": round(f1_weighted, 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "total_samples": int(len(y_true))
    }
    return metrics


def perform_error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    max_samples: int = 15
) -> List[Dict[str, Any]]:
    """
    Performs error analysis by finding misclassified customer reviews.
    Extracts review text, true label, predicted label, and prediction confidence if available.
    """
    errors = []
    text_col = "review_text" if "review_text" in df.columns else "full_review"

    for idx, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        if true_val != pred_val:
            row_data = df.iloc[idx]
            error_item = {
                "index": int(idx),
                "review_text": str(row_data.get(text_col, "")),
                "actual_sentiment": str(true_val),
                "predicted_sentiment": str(pred_val),
                "rating": int(row_data.get("rating", 0)) if "rating" in row_data else None,
                "department": str(row_data.get("department_name", "Unknown"))
            }
            if y_probs is not None:
                max_prob = float(np.max(y_probs[idx]))
                error_item["confidence"] = round(max_prob, 4)
            errors.append(error_item)

            if len(errors) >= max_samples:
                break

    return errors


def save_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """Saves model metadata and performance reports to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", filepath)
