from __future__ import annotations
import numpy as np


def normalized_categorical_entropy(probs, axis=-1, eps=1e-12):
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    k = p.shape[axis]
    h = -(p * np.log(p)).sum(axis=axis)
    return h / np.log(k)


def binary_entropy(prob, eps=1e-12):
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / np.log(2.0)
