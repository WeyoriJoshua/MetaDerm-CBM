from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss,
)


def expected_calibration_error(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not mask.any(): continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true, prob, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity_recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "brier": brier_score_loss(y, p),
        "ece_10": expected_calibration_error(y, p, 10),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k,v in out.items()}
