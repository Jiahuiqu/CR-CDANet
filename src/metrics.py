"""Binary change-detection metrics: OA, Kappa, F1, precision and recall.

Metrics whose denominator can vanish -- precision and recall when nothing (or
everything) is predicted as changed, F1 when both are zero, Kappa for a
degenerate label distribution -- return ``0.0`` in that case instead of raising.
"""

import numpy as np


def calculate_confusion_matrix(preds, labels):
    """Return ``TP, FP, TN, FN`` for the changed class."""
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    return float(tp), float(fp), float(tn), float(fn)


def calculate_OA(tp, fp, tn, fn):
    """Overall accuracy."""
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


def calculate_Kappa(tp, fp, tn, fn):
    """Cohen's kappa coefficient."""
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    po = (tp + tn) / total
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (total * total)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def calculate_Pr(tp, fp):
    """Precision."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_Re(tp, fn):
    """Recall."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_F1(pr, re):
    """F1 score."""
    return 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0


def calculate_AA(tp, fp, tn, fn):
    """Average accuracy of the two classes."""
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.
    return (tpr + tnr) / 2


def calculate_metrics(preds, labels):
    """Return ``(OA, Kappa, F1, Precision, Recall)``."""
    tp, fp, tn, fn = calculate_confusion_matrix(preds, labels)

    oa = calculate_OA(tp, fp, tn, fn)
    kappa = calculate_Kappa(tp, fp, tn, fn)
    pr = calculate_Pr(tp, fp)
    re = calculate_Re(tp, fn)
    f1 = calculate_F1(pr, re)
    return oa, kappa, f1, pr, re


def format_metrics(oa, kappa, f1, pr, re):
    return (f"OA: {oa:.4f}  KC: {kappa:.4f}  F1: {f1:.4f}  "
            f"Pr: {pr:.4f}  Re: {re:.4f}")
