from __future__ import annotations
import numpy as np
from ..constants import CONCEPT_COLUMNS
from ..uncertainty.entropy import normalized_categorical_entropy


def single_sample_dict(concept_probs, i):
    return {c: np.asarray(concept_probs[c][i:i+1]).copy() for c in CONCEPT_COLUMNS}


def reasoning_impact(reasoner, concept_probs, sample_index: int, concept: str):
    base = single_sample_dict(concept_probs, sample_index)
    values = []
    k = base[concept].shape[1]
    for state in range(k):
        modified = {c: p.copy() for c, p in base.items()}
        modified[concept][:] = 0.0
        modified[concept][0, state] = 1.0
        values.append(float(reasoner.predict(modified)["probability"][0]))
    return max(values) - min(values)


def select_query_concept(reasoner, concept_probs, sample_index: int):
    scored = []
    for c in CONCEPT_COLUMNS:
        uncertainty = float(normalized_categorical_entropy(concept_probs[c][sample_index:sample_index+1])[0])
        impact = reasoning_impact(reasoner, concept_probs, sample_index, c)
        scored.append((uncertainty * impact, c, uncertainty, impact))
    scored.sort(reverse=True)
    return scored[0], scored


def apply_oracle_correction(concept_probs, sample_index: int, concept: str, true_state: int):
    corrected = single_sample_dict(concept_probs, sample_index)
    corrected[concept][0, :] = 0.0
    corrected[concept][0, int(true_state)] = 1.0
    return corrected
