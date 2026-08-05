"""Metrics computation module (Official GLUE, Diagnostics, ECE Calibration, Brier Score, Failure Analysis)."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, brier_score_loss


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * (bin_size / len(labels))

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Compute Brier Score for multi-class or binary predictions."""
    if num_classes == 2:
        # Binary Brier Score on positive class probability
        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return float(brier_score_loss(labels, pos_probs))
    else:
        # Multi-class Brier score
        one_hot = np.eye(num_classes)[labels]
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    raw_texts: List[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute official metrics, diagnostic metrics, calibration scores, predictions df, and misclassified df.

    Args:
        logits: Model output logits shape (N, num_classes).
        labels: Ground truth class indices shape (N,).
        dataset_name: Name of GLUE task ('sst2', 'mrpc', 'rte').
        raw_texts: List of raw input text strings corresponding to validation items.

    Returns:
        Tuple of (metrics_dict, predictions_df, misclassified_df).
    """
    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    acc = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    # Task official metrics specification
    official_metrics = {}
    if dataset_name == "mrpc":
        # MRPC official evaluation: Accuracy and binary F1 for paraphrase class
        _, _, binary_f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        official_metrics["accuracy"] = float(acc)
        official_metrics["f1"] = float(binary_f1)
    else:  # sst2, rte
        official_metrics["accuracy"] = float(acc)

    num_classes = logits.shape[1]
    ece = compute_expected_calibration_error(probs, labels, num_bins=10)
    brier = compute_brier_score(probs, labels, num_classes=num_classes)

    diagnostic_metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1),
        "expected_calibration_error": float(ece),
        "brier_score": float(brier),
        "confusion_matrix": cm
    }

    metrics_dict = {
        "official_metrics": official_metrics,
        "diagnostic_metrics": diagnostic_metrics
    }

    # Dataframe for predictions.csv
    pred_df_data = {
        "id": list(range(len(labels))),
        "label": labels.tolist(),
        "prediction": preds.tolist(),
        "probability": max_probs.tolist()
    }
    predictions_df = pd.DataFrame(pred_df_data)

    # Dataframe for misclassified.csv
    misclassified_mask = preds != labels
    misclassified_data = {
        "id": np.where(misclassified_mask)[0].tolist(),
        "label": labels[misclassified_mask].tolist(),
        "prediction": preds[misclassified_mask].tolist(),
        "probability": max_probs[misclassified_mask].tolist()
    }
    if raw_texts and len(raw_texts) == len(labels):
        misclassified_data["input_text"] = [raw_texts[i] for i in np.where(misclassified_mask)[0]]
    else:
        misclassified_data["input_text"] = ["N/A"] * int(np.sum(misclassified_mask))

    misclassified_df = pd.DataFrame(misclassified_data)

    return metrics_dict, predictions_df, misclassified_df
