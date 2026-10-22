from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from ..constants import CONCEPT_COLUMNS


def balanced_class_weights(labels: np.ndarray, n_classes: int, clip_max: float = 10.0):
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (n_classes * counts)
    weights = np.minimum(weights, clip_max)
    return torch.tensor(weights, dtype=torch.float32)


def concept_loss(logits, targets, class_weights=None, label_smoothing=0.0):
    losses = {}
    for c in CONCEPT_COLUMNS:
        weight = class_weights[c].to(logits[c].device) if class_weights else None
        losses[c] = F.cross_entropy(logits[c], targets[c], weight=weight, label_smoothing=label_smoothing)
    total = torch.stack(list(losses.values())).mean()
    return total, losses


def binary_pos_weight(labels: np.ndarray):
    pos = max(float(labels.sum()), 1.0)
    neg = max(float(len(labels) - labels.sum()), 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)
