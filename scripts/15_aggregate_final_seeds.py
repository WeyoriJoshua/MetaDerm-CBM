from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _seed_from_name(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _mean_sd(df, group_cols, exclude=()):
    numeric = [
        c for c in df.columns
        if c not in set(group_cols) | set(exclude)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    out = (
        df.groupby(group_cols)[numeric]
        .agg(["mean", "std"])
    )
    out.columns = [
        f"{a}_{b}" for a, b in out.columns
    ]
    return out.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        default="results/efficientnet_b3_final",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    out = root / "publication" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # Diagnostic/selective results.
    diagnostic_rows = []
    for seed_dir in sorted(
        (root / "final").glob("seed*")
    ):
        metrics_file = seed_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        seed = _seed_from_name(seed_dir.name)
        m = json.loads(
            metrics_file.read_text(
                encoding="utf-8"
            )
        )

        for model in [
            "blackbox",
            "soft_reasoner",
        ]:
            if model not in m:
                continue
            diagnostic_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    **m[model],
                    "predict_coverage": m.get(
                        "coverage_predict"
                    ),
                    "query_rate": m.get(
                        "query_rate"
                    ),
                    "abstain_rate": m.get(
                        "abstain_rate"
                    ),
                    "tau_low": m.get(
                        "tau_low"
                    ),
                    "tau_high": m.get(
                        "tau_high"
                    ),
                }
            )

    if diagnostic_rows:
        d = pd.DataFrame(diagnostic_rows)
        d.to_csv(
            out / "diagnostic_results_per_seed.csv",
            index=False,
        )
        s = _mean_sd(
            d,
            ["model"],
            exclude=("seed",),
        )
        s.to_csv(
            out / "diagnostic_results_mean_sd.csv",
            index=False,
        )

    # Concept results.
    concept_rows = []
    for seed_dir in sorted(
        (root / "final").glob("seed*")
    ):
        f = seed_dir / "concept_metrics.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.insert(
            0,
            "seed",
            _seed_from_name(seed_dir.name),
        )
        concept_rows.append(d)

    if concept_rows:
        cdf = pd.concat(
            concept_rows,
            ignore_index=True,
        )
        cdf.to_csv(
            out / "concept_results_per_seed.csv",
            index=False,
        )
        csum = _mean_sd(
            cdf,
            ["concept"],
            exclude=("seed",),
        )
        csum.to_csv(
            out / "concept_results_mean_sd.csv",
            index=False,
        )

    # CBM baselines.
    cbm_rows = []
    for seed_dir in sorted(
        (root / "cbm_baselines").glob("seed*")
    ):
        f = seed_dir / "metrics.json"
        if not f.exists():
            continue
        seed = _seed_from_name(seed_dir.name)
        m = json.loads(
            f.read_text(encoding="utf-8")
        )
        for model in [
            "predicted_concept_cbm",
            "oracle_concept_cbm",
        ]:
            if model in m:
                cbm_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        **m[model],
                    }
                )

    if cbm_rows:
        b = pd.DataFrame(cbm_rows)
        b.to_csv(
            out / "cbm_results_per_seed.csv",
            index=False,
        )
        bsum = _mean_sd(
            b,
            ["model"],
            exclude=("seed",),
        )
        bsum.to_csv(
            out / "cbm_results_mean_sd.csv",
            index=False,
        )

    # Final intervention results.
    intervention_rows = []
    for seed_dir in sorted(
        (root / "intervention_final").glob(
            "seed*"
        )
    ):
        f = seed_dir / "summary.json"
        if not f.exists():
            continue
        m = json.loads(
            f.read_text(encoding="utf-8")
        )
        before = m.get(
            "reasoner_before",
            {},
        )
        after = m.get(
            "reasoner_after",
            {},
        )
        intervention_rows.append(
            {
                "seed": m.get(
                    "seed",
                    _seed_from_name(seed_dir.name),
                ),
                "n_query": m.get("n_query"),
                "query_rate": m.get(
                    "query_rate"
                ),
                "correction_rate": m.get(
                    "selected_concept_correction_rate"
                ),
                "reason_accuracy_before": before.get(
                    "accuracy"
                ),
                "reason_accuracy_after": after.get(
                    "accuracy"
                ),
                "accuracy_gain": m.get(
                    "absolute_reason_accuracy_gain"
                ),
                "net_error_reduction": m.get(
                    "net_error_reduction"
                ),
                "mean_reliability_before": m.get(
                    "mean_reliability_before"
                ),
                "mean_reliability_after": m.get(
                    "mean_reliability_after"
                ),
            }
        )

    if intervention_rows:
        q = pd.DataFrame(
            intervention_rows
        )
        q.to_csv(
            out / "intervention_per_seed.csv",
            index=False,
        )

        numeric = [
            c for c in q.columns
            if c != "seed"
        ]
        summary = pd.DataFrame(
            {
                "metric": numeric,
                "mean": [
                    q[c].mean()
                    for c in numeric
                ],
                "std": [
                    q[c].std(ddof=1)
                    for c in numeric
                ],
            }
        )
        summary.to_csv(
            out / "intervention_mean_sd.csv",
            index=False,
        )

    print("final publication tables:", out)


if __name__ == "__main__":
    main()
