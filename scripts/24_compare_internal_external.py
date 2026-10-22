
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description=(
            "Compare frozen Derm7pt internal results "
            "with PH2 external validation."
        )
    )
    p.add_argument(
        "--internal-root",
        default="results/efficientnet_b3_final",
    )
    p.add_argument(
        "--external-root",
        default=(
            "results/ph2_external/"
            "efficientnet_b3"
        ),
    )
    args = p.parse_args()

    iroot = Path(args.internal_root)
    eroot = Path(args.external_root)

    internal = pd.read_csv(
        iroot
        / "publication"
        / "tables"
        / "diagnostic_results_per_seed.csv"
    )
    external = pd.read_csv(
        eroot
        / "diagnostic_per_seed.csv"
    )

    rows = []
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "brier",
        "ece_10",
    ]

    for model in [
        "blackbox",
        "soft_reasoner",
    ]:
        a = internal[
            internal["model"] == model
        ].set_index("seed")
        b = external[
            external["model"] == model
        ].set_index("seed")

        common = a.index.intersection(
            b.index
        )

        for metric in metrics:
            delta = (
                b.loc[common, metric]
                - a.loc[common, metric]
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "internal_mean": a.loc[
                        common, metric
                    ].mean(),
                    "external_mean": b.loc[
                        common, metric
                    ].mean(),
                    "external_minus_internal": (
                        delta.mean()
                    ),
                    "delta_sd": delta.std(
                        ddof=1
                    ),
                }
            )

    result = pd.DataFrame(rows)
    out = (
        eroot
        / "internal_external_comparison.csv"
    )
    result.to_csv(
        out,
        index=False,
    )

    print(result.to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
