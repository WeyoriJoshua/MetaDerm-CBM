from __future__ import annotations
import numpy as np
from ..constants import CONCEPT_COLUMNS


class SoftSevenPointReasoner:
    """Research implementation of a soft seven-point checklist pathway.

    The suspicious state sets are configuration-controlled because operational mappings
    (especially vascular structures) must be declared explicitly in the manuscript.
    """
    def __init__(self, cfg):
        self.vocab = cfg["data"]["concepts"]
        self.criteria = cfg["reasoning"]["criteria"]
        self.threshold = float(cfg["reasoning"].get("threshold", 3.0))
        self.temperature = float(cfg["reasoning"].get("temperature", 0.75))

    def suspicious_probability(self, concept: str, probs):
        states = self.vocab[concept]
        positive = set(self.criteria[concept]["suspicious_states"])
        idx = [i for i, s in enumerate(states) if s in positive]
        if not idx:
            return np.zeros(len(probs), dtype=float)
        return np.asarray(probs)[:, idx].sum(axis=1)

    def score(self, concept_probs: dict[str, np.ndarray]):
        n = len(next(iter(concept_probs.values())))
        score = np.zeros(n, dtype=float)
        components = {}
        for c in CONCEPT_COLUMNS:
            p = self.suspicious_probability(c, concept_probs[c])
            contribution = float(self.criteria[c]["weight"]) * p
            components[c] = contribution
            score += contribution
        return score, components

    def probability_from_score(self, score):
        score = np.asarray(score, dtype=float)
        return 1.0 / (1.0 + np.exp(-(score - self.threshold) / self.temperature))

    def predict(self, concept_probs):
        score, components = self.score(concept_probs)
        return {"score": score, "probability": self.probability_from_score(score), "components": components}
