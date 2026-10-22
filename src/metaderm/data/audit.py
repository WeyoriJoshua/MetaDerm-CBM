from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image
from ..constants import CONCEPT_COLUMNS
from .derm7pt import load_metadata, load_split_indices, resolve_paths, validate_concept_vocab
from ..reproducibility import sha256_file


def profile_key(row, cfg):
    vocab = cfg["data"]["concepts"]
    maps = {c: {v: i for i, v in enumerate(vocab[c])} for c in CONCEPT_COLUMNS}
    return tuple(maps[c][str(row[c])] for c in CONCEPT_COLUMNS)


def profile_reliability_table(train_df: pd.DataFrame, cfg) -> pd.DataFrame:
    groups = defaultdict(lambda: [0, 0])
    for _, row in train_df.iterrows():
        k = profile_key(row, cfg)
        y = int(row["binary_label"])
        groups[k][y] += 1
    rows = []
    for k, (n0, n1) in groups.items():
        n = n0 + n1
        p1 = n1 / n
        purity = max(n0, n1) / n
        reliability = 2.0 * (purity - 0.5)
        rows.append({
            "profile": "|".join(map(str, k)), "n": n, "n_non_melanoma": n0, "n_melanoma": n1,
            "melanoma_rate": p1, "purity": purity, "reliability": reliability,
            "inconsistent": int(n0 > 0 and n1 > 0),
        })
    return pd.DataFrame(rows).sort_values(["inconsistent", "n"], ascending=[False, False])


def audit_dataset(cfg, check_images=True):
    df = load_metadata(cfg)
    validate_concept_vocab(df, cfg)
    paths = resolve_paths(cfg)
    splits = {s: load_split_indices(cfg, s) for s in ["train", "valid", "test"]}
    split_sets = {k: set(v) for k, v in splits.items()}
    overlaps = {
        "train_valid": len(split_sets["train"] & split_sets["valid"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "valid_test": len(split_sets["valid"] & split_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Split leakage detected: {overlaps}")

    image_errors = []
    if check_images:
        for i, row in df.iterrows():
            path = paths["images"] / str(row[cfg["data"]["image_col"]])
            if not path.exists():
                image_errors.append({"index": i, "path": str(path), "error": "missing"})
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as e:
                image_errors.append({"index": i, "path": str(path), "error": str(e)})

    split_summary = []
    concept_summary = []
    for split, idx in splits.items():
        part = df.iloc[idx]
        split_summary.append({
            "split": split, "n": len(part), "melanoma_n": int(part["binary_label"].sum()),
            "melanoma_prevalence": float(part["binary_label"].mean()),
        })
        for c in CONCEPT_COLUMNS:
            for state, n in part[c].value_counts(dropna=False).items():
                concept_summary.append({"split": split, "concept": c, "state": state, "n": int(n)})

    train_df = df.iloc[splits["train"]].copy()
    profile_df = profile_reliability_table(train_df, cfg)
    fingerprint = {
        "meta_sha256": sha256_file(paths["meta"]),
        **{f"{s}_split_sha256": sha256_file(paths[f"split_{s}"]) for s in splits},
    }
    return {
        "split_summary": pd.DataFrame(split_summary),
        "concept_summary": pd.DataFrame(concept_summary),
        "profiles": profile_df,
        "image_errors": pd.DataFrame(image_errors),
        "overlaps": overlaps,
        "fingerprint": fingerprint,
        "n_total": len(df),
    }
