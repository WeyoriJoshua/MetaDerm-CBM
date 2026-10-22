from __future__ import annotations
import numpy as np
from ..constants import CONCEPT_COLUMNS
from ..uncertainty.entropy import normalized_categorical_entropy, binary_entropy


def metacognitive_features(concept_probs, blackbox_prob, reason_prob, profile_reliability):
    entropies = np.stack([
        normalized_categorical_entropy(concept_probs[c]) for c in CONCEPT_COLUMNS
    ], axis=1)
    mean_concept_uncertainty = entropies.mean(axis=1)
    diagnosis_uncertainty = binary_entropy(blackbox_prob)
    disagreement = np.abs(np.asarray(blackbox_prob) - np.asarray(reason_prob))
    return {
        "concept_uncertainty": mean_concept_uncertainty,
        "concept_confidence": 1 - mean_concept_uncertainty,
        "diagnosis_uncertainty": diagnosis_uncertainty,
        "diagnosis_confidence": 1 - diagnosis_uncertainty,
        "neural_symbolic_disagreement": disagreement,
        "neural_symbolic_agreement": 1 - disagreement,
        "profile_reliability": np.asarray(profile_reliability, dtype=float),
        "per_concept_entropy": entropies,
    }


def reliability_score(features, cfg):
    w = cfg["metacognition"]["weights"]
    terms = {
        "concept_confidence": features["concept_confidence"],
        "diagnosis_confidence": features["diagnosis_confidence"],
        "neural_symbolic_agreement": features["neural_symbolic_agreement"],
        "profile_reliability": features["profile_reliability"],
    }
    denom = sum(float(w.get(k, 0.0)) for k in terms)
    if denom <= 0:
        raise ValueError("Metacognition weights must sum to > 0")
    score = sum(float(w.get(k, 0.0)) * v for k, v in terms.items()) / denom
    return np.clip(score, 0.0, 1.0)


def risk_coverage_curve(y_true, y_pred, reliability):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    reliability = np.asarray(reliability)
    order = np.argsort(-reliability)
    correct = (y_true[order] == y_pred[order]).astype(float)
    cum_errors = np.cumsum(1 - correct)
    n = len(y_true)
    rows = []
    for k in range(1, n + 1):
        coverage = k / n
        risk = cum_errors[k - 1] / k
        rows.append((coverage, risk, reliability[order[k - 1]]))
    return np.asarray(rows, dtype=float)


def select_three_way_thresholds(y_true, y_pred, reliability, target_risk=0.10, minimum_coverage=0.50, abstain_fraction_of_deferred=0.35):
    curve = risk_coverage_curve(y_true, y_pred, reliability)
    candidates = curve[(curve[:, 1] <= target_risk) & (curve[:, 0] >= minimum_coverage)]
    if len(candidates):
        best = candidates[np.argmax(candidates[:, 0])]
        tau_high = float(best[2])
    else:
        tau_high = float(np.quantile(reliability, 0.60))
    deferred = reliability[reliability < tau_high]
    if len(deferred) == 0:
        tau_low = tau_high
    else:
        tau_low = float(np.quantile(deferred, abstain_fraction_of_deferred))
    return tau_low, tau_high


def assign_actions(reliability, tau_low, tau_high):
    r = np.asarray(reliability)
    out = np.full(r.shape, "query", dtype=object)
    out[r >= tau_high] = "predict"
    out[r <= tau_low] = "abstain"
    return out
