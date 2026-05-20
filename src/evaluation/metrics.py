"""
Evaluation metrics for chest X-ray classification.

Includes standard classification metrics plus calibration analysis
(Expected Calibration Error, reliability diagrams) which is critical
for clinical deployment but often overlooked.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from typing import Dict, Tuple, List, Optional


class ClinicalMetrics:
    """
    Comprehensive evaluation for binary medical classification.

    Computes:
        - Accuracy, sensitivity, specificity, PPV, NPV
        - F1, MCC, AUC
        - Confusion matrix values (TP, TN, FP, FN)
        - Expected Calibration Error (ECE)
        - Reliability diagram bins
    """

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.results_ = None

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Compute all metrics.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_proba: Predicted probabilities for the positive class

        Returns:
            Dictionary of all metric values
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = precision_score(y_true, y_pred, zero_division=0)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = None

        ece, reliability = self._calibration_metrics(y_true, y_proba)

        results = {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "f1_score": float(f1),
            "mcc": float(mcc),
            "auc": auc,
            "ece": float(ece),
            "reliability_bins": reliability,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        self.results_ = results
        return results

    def _calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Tuple[float, List[Dict]]:
        """
        Compute Expected Calibration Error and reliability bins.

        ECE measures the gap between confidence and accuracy across
        probability bins. A well-calibrated model has ECE close to 0.

        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (ECE value, list of bin statistics)
        """
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]

        ece = 0.0
        reliability = []
        total = len(y_proba)

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > lower) & (y_proba <= upper)
            bin_size = int(in_bin.sum())

            if bin_size == 0:
                reliability.append({
                    "lower": float(lower),
                    "upper": float(upper),
                    "size": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                })
                continue

            avg_confidence = float(y_proba[in_bin].mean())
            accuracy = float((y_true[in_bin] == 1).mean())
            gap = abs(avg_confidence - accuracy)

            ece += (bin_size / total) * gap

            reliability.append({
                "lower": float(lower),
                "upper": float(upper),
                "size": bin_size,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "gap": float(gap),
            })

        return float(ece), reliability

    def summary_table(self) -> str:
        """Return formatted summary of metrics."""
        if self.results_ is None:
            return "No results computed. Call compute() first."

        r = self.results_
        lines = [
            "=" * 55,
            "  Chest X-Ray Pneumonia — Evaluation Summary",
            "=" * 55,
            f"  Accuracy           : {r['accuracy']:.4f}",
            f"  Sensitivity (TPR)  : {r['sensitivity']:.4f}",
            f"  Specificity (TNR)  : {r['specificity']:.4f}",
            f"  PPV (Precision)    : {r['ppv']:.4f}",
            f"  NPV                : {r['npv']:.4f}",
            f"  F1-Score           : {r['f1_score']:.4f}",
            f"  MCC                : {r['mcc']:.4f}",
        ]
        if r['auc'] is not None:
            lines.append(f"  AUC                : {r['auc']:.4f}")
        lines.extend([
            "  -- Calibration --",
            f"  ECE ({self.num_bins} bins)        : {r['ece']:.4f}",
            "-" * 55,
            f"  TP: {r['true_positives']}  TN: {r['true_negatives']}  "
            f"FP: {r['false_positives']}  FN: {r['false_negatives']}",
            "=" * 55,
        ])
        return "\n".join(lines)


if __name__ == "__main__":
    np.random.seed(42)

    n = 500
    y_true = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    y_proba = np.where(
        y_true == 1,
        np.clip(0.7 + 0.2 * np.random.randn(n), 0, 1),
        np.clip(0.3 + 0.2 * np.random.randn(n), 0, 1),
    )
    y_pred = (y_proba > 0.5).astype(int)

    evaluator = ClinicalMetrics(num_bins=10)
    evaluator.compute(y_true, y_pred, y_proba)
    print(evaluator.summary_table())
