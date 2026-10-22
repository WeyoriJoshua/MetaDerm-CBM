from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from ..constants import CONCEPT_COLUMNS


def _safe_macro_ovr_auc(y, p):
    """
    Compute macro one-vs-rest AUROC robustly.

    Classes absent from the current evaluation split are skipped,
    because ROC AUC is undefined when only one binary class is present.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    aucs = []

    for class_index in range(p.shape[1]):
        y_binary = (y == class_index).astype(int)

        # ROC AUC requires both positive and negative examples.
        if np.unique(y_binary).size < 2:
            continue

        auc = roc_auc_score(
            y_binary,
            p[:, class_index],
        )

        aucs.append(float(auc))

    if not aucs:
        return float("nan")

    return float(np.mean(aucs))


def concept_metrics(targets, probs, cfg):
    """
    Evaluate the seven Derm7pt concept heads.
    """
    rows = []

    for concept in CONCEPT_COLUMNS:
        y = np.asarray(
            targets[concept],
            dtype=int,
        )

        p = np.asarray(
            probs[concept],
            dtype=float,
        )

        pred = np.argmax(
            p,
            axis=1,
        )

        rows.append(
            {
                "concept": concept,
                "accuracy": float(
                    accuracy_score(y, pred)
                ),
                "macro_f1": float(
                    f1_score(
                        y,
                        pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "macro_ovr_auroc": float(
                    _safe_macro_ovr_auc(y, p)
                ),
            }
        )

    df = pd.DataFrame(rows)

    valid_auc = df["macro_ovr_auroc"].dropna()

    summary = {
        "mean_accuracy": float(
            df["accuracy"].mean()
        ),
        "mean_macro_f1": float(
            df["macro_f1"].mean()
        ),
        "mean_macro_ovr_auroc": (
            float(valid_auc.mean())
            if not valid_auc.empty
            else float("nan")
        ),
    }

    return df, summary
