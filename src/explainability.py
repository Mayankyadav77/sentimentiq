"""
explainability.py - Model-Specific Sentiment Prediction Explainer
-----------------------------------------------------------------
Platform: SentimentIQ – Customer Review Intelligence Platform

This module implements model-specific feature attribution for:
1. Logistic Regression: Log-odds feature contributions (w_j * x_j).
2. Linear SVM: Margin decision score contributions (w_j * x_j).
3. Multinomial Naive Bayes: Relative log-likelihood contributions.

Key Capabilities:
- Identifies top words driving the prediction toward the predicted sentiment.
- Identifies words counteracting or opposing the predicted sentiment.
- Generates transparent, human-readable explanations without fabricating causal claims.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger("Explainability")


class SentimentExplainer:
    """
    Interprets predictions of linear and probabilistic text classifiers.
    Calculates exact token-level contributions to the decision score.
    """

    def __init__(self, model: Any, feature_names: np.ndarray, class_labels: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.class_labels = list(class_labels)
        self.model_type = type(model).__name__

    def explain_instance(
        self,
        X_vec: csr_matrix,
        predicted_class: str,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Calculates exact word contributions for a single review vector X_vec.

        Parameters:
        -----------
        X_vec : csr_matrix
            Single-row sparse matrix of TF-IDF feature weights.
        predicted_class : str
            The predicted sentiment class ('Negative', 'Neutral', or 'Positive').
        top_k : int
            Number of top contributing words to return.

        Returns:
        --------
        Dict with keys:
            - 'predicted_class': str
            - 'supporting_words': List[Dict[str, Any]] (token, score)
            - 'opposing_words': List[Dict[str, Any]] (token, score)
            - 'summary_text': str
            - 'disclaimer': str
        """
        if X_vec.nnz == 0:
            return {
                "predicted_class": predicted_class,
                "supporting_words": [],
                "opposing_words": [],
                "summary_text": "No known vocabulary terms detected in the review to explain.",
                "disclaimer": "Feature attribution requires recognized vocabulary tokens."
            }

        if predicted_class not in self.class_labels:
            class_idx = 0
        else:
            class_idx = self.class_labels.index(predicted_class)

        # Extract non-zero column indices and TF-IDF values for this document
        row = X_vec.getrow(0)
        feature_indices = row.indices
        feature_values = row.data

        contributions: List[Tuple[str, float]] = []

        if hasattr(self.model, "coef_"):
            # Linear Models: LogisticRegression & LinearSVC
            coefs = self.model.coef_
            
            # Handle binary vs multiclass shape
            if coefs.ndim == 1 or coefs.shape[0] == 1:
                # Binary classification
                target_weights = coefs[0] if class_idx == 1 else -coefs[0]
            else:
                # Multiclass (one-vs-rest / multinomial)
                target_weights = coefs[class_idx]

            for idx, val in zip(feature_indices, feature_values):
                token = self.feature_names[idx]
                contrib = float(target_weights[idx] * val)
                contributions.append((token, contrib))

        elif hasattr(self.model, "feature_log_prob_"):
            # Probabilistic: Multinomial Naive Bayes
            log_probs = self.model.feature_log_prob_
            # Relative log-likelihood difference compared to the average across other classes
            mean_other_log_probs = np.mean(
                [log_probs[c] for c in range(len(self.class_labels)) if c != class_idx],
                axis=0
            )
            relative_weights = log_probs[class_idx] - mean_other_log_probs

            for idx, val in zip(feature_indices, feature_values):
                token = self.feature_names[idx]
                contrib = float(relative_weights[idx] * val)
                contributions.append((token, contrib))
        else:
            # Fallback based strictly on TF-IDF term salience
            for idx, val in zip(feature_indices, feature_values):
                token = self.feature_names[idx]
                contributions.append((token, float(val)))

        # Separate into supporting (positive contribution) and opposing (negative contribution)
        supporting = [(tok, round(c, 4)) for tok, c in contributions if c > 0]
        opposing = [(tok, round(c, 4)) for tok, c in contributions if c < 0]

        # Sort supporting descending (highest positive first)
        supporting.sort(key=lambda x: x[1], reverse=True)
        # Sort opposing ascending (most negative first)
        opposing.sort(key=lambda x: x[1])

        top_supporting = supporting[:top_k]
        top_opposing = opposing[:top_k]

        # Generate human-readable narrative explanation
        if top_supporting:
            top_words_str = ", ".join([f"'{w}' (+{s:.3f})" for w, s in top_supporting[:3]])
            summary = (
                f"Prediction '{predicted_class}' was strongly influenced by terms: {top_words_str}."
            )
            if top_opposing:
                opp_words_str = ", ".join([f"'{w}' ({s:.3f})" for w, s in top_opposing[:2]])
                summary += f" Counteracting signals were detected from: {opp_words_str}."
        else:
            summary = f"Prediction '{predicted_class}' was determined primarily by base model bias and prior distribution."

        return {
            "predicted_class": predicted_class,
            "supporting_words": [{"word": w, "weight": s} for w, s in top_supporting],
            "opposing_words": [{"word": w, "weight": s} for w, s in top_opposing],
            "summary_text": summary,
            "disclaimer": (
                "Note: Feature attributions represent linear model weights in the learned TF-IDF space, "
                "not causal relationships in natural language."
            )
        }
