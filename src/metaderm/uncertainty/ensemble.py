from __future__ import annotations
import numpy as np


def ensemble_mean(probs_list):
    arr = np.stack(probs_list, axis=0)
    return arr.mean(axis=0)


def ensemble_variance(probs_list):
    arr = np.stack(probs_list, axis=0)
    return arr.var(axis=0)
