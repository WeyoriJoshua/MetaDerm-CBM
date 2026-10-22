from __future__ import annotations
from collections import defaultdict
import numpy as np
from ..constants import CONCEPT_COLUMNS


def hard_profile_from_probs(concept_probs: dict[str, np.ndarray]):
    arrs = [np.argmax(concept_probs[c], axis=1) for c in CONCEPT_COLUMNS]
    return list(zip(*arrs))


def build_profile_map(train_df, cfg):
    vocab = cfg["data"]["concepts"]
    maps = {c: {v: i for i, v in enumerate(vocab[c])} for c in CONCEPT_COLUMNS}
    counts = defaultdict(lambda: [0, 0])
    for _, row in train_df.iterrows():
        profile = tuple(maps[c][str(row[c])] for c in CONCEPT_COLUMNS)
        counts[profile][int(row["binary_label"])] += 1
    out = {}
    for profile, (n0, n1) in counts.items():
        total = n0 + n1
        purity = max(n0, n1) / total
        out[profile] = {
            "n": total, "n0": n0, "n1": n1,
            "melanoma_rate": n1 / total,
            "reliability": 2 * (purity - 0.5),
            "inconsistent": bool(n0 and n1),
        }
    return out


def lookup_profile_reliability(profiles, profile_map, unseen=0.0):
    return np.array([profile_map.get(tuple(p), {}).get("reliability", unseen) for p in profiles], dtype=float)
