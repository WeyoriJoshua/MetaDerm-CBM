from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci(values, n_boot=10000, seed=2026):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return (
        float(x.mean()),
        float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def savefig(fig, outbase: Path):
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        default="results/efficientnet_b3_final",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=10000,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    args = p.parse_args()

    root = Path(args.results_root)
    tables = root / "publication" / "tables"
    figures = root / "publication" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    selective = pd.read_csv(
        tables / "selective_performance_per_seed.csv"
    )

    # ------------------------------------------------------------------
    # Bootstrap CI table for selective metrics
    # ------------------------------------------------------------------
    metrics = [
        "predict_coverage",
        "query_rate",
        "abstain_rate",
        "selective_risk",
        "accepted_accuracy",
        "accepted_balanced_accuracy",
        "accepted_sensitivity",
        "accepted_specificity",
        "benign_predict_coverage",
        "melanoma_predict_coverage",
        "error_capture_rate",
    ]

    rows = []
    for metric in metrics:
        mean, sd, lo, hi = bootstrap_ci(
            selective[metric].to_numpy(),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "min": float(selective[metric].min()),
                "max": float(selective[metric].max()),
            }
        )

    bootstrap_df = pd.DataFrame(rows)
    bootstrap_df.to_csv(
        tables / "selective_performance_bootstrap_ci.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Table: model performance across seeds
    # ------------------------------------------------------------------
    diag = pd.read_csv(
        tables / "diagnostic_results_per_seed.csv"
    )
    model_metrics = [
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

    model_rows = []
    for model, g in diag.groupby("model"):
        for metric in model_metrics:
            mean, sd, lo, hi = bootstrap_ci(
                g[metric].to_numpy(),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            model_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )

    pd.DataFrame(model_rows).to_csv(
        tables / "model_performance_bootstrap_ci.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Table: intervention reproducibility
    # ------------------------------------------------------------------
    intervention_file = tables / "intervention_per_seed.csv"
    if intervention_file.exists():
        intervention = pd.read_csv(intervention_file)
        intervention_metrics = [
            "correction_rate",
            "reason_accuracy_before",
            "reason_accuracy_after",
            "accuracy_gain",
            "net_error_reduction",
            "mean_reliability_before",
            "mean_reliability_after",
        ]
        ir = []
        for metric in intervention_metrics:
            mean, sd, lo, hi = bootstrap_ci(
                intervention[metric].to_numpy(),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            ir.append(
                {
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )
        pd.DataFrame(ir).to_csv(
            tables / "intervention_bootstrap_ci.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Figure 1: selective risk vs coverage
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        selective["predict_coverage"],
        selective["selective_risk"],
        s=70,
    )
    for _, row in selective.iterrows():
        ax.annotate(
            str(int(row["seed"])),
            (
                row["predict_coverage"],
                row["selective_risk"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Autonomous PREDICT coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Selective risk–coverage across training seeds")
    ax.grid(alpha=0.25)
    savefig(fig, figures / "Fig_selective_risk_coverage")

    # ------------------------------------------------------------------
    # Figure 2: benign vs melanoma predict coverage
    # ------------------------------------------------------------------
    x = np.arange(len(selective))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.bar(
        x - width / 2,
        selective["benign_predict_coverage"],
        width,
        label="Benign",
    )
    ax.bar(
        x + width / 2,
        selective["melanoma_predict_coverage"],
        width,
        label="Melanoma",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(selective["seed"].astype(str))
    ax.set_xlabel("Training seed")
    ax.set_ylabel("PREDICT coverage")
    ax.set_title("Class-conditional autonomous coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, figures / "Fig_class_conditional_coverage")

    # ------------------------------------------------------------------
    # Figure 3: error capture
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.bar(
        selective["seed"].astype(str),
        selective["error_capture_rate"],
    )
    ax.axhline(
        selective["error_capture_rate"].mean(),
        linestyle="--",
        linewidth=1.5,
        label="Mean",
    )
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Fraction of black-box errors deferred")
    ax.set_ylim(0, 1.0)
    ax.set_title("Error capture by QUERY/ABSTAIN")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, figures / "Fig_error_capture")

    # ------------------------------------------------------------------
    # Figure 4: intervention before/after
    # ------------------------------------------------------------------
    if intervention_file.exists():
        intervention = pd.read_csv(intervention_file)
        x = np.arange(len(intervention))
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        ax.plot(
            x,
            intervention["reason_accuracy_before"],
            marker="o",
            label="Before intervention",
        )
        ax.plot(
            x,
            intervention["reason_accuracy_after"],
            marker="o",
            label="After intervention",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(intervention["seed"].astype(str))
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Reasoner accuracy on QUERY cases")
        ax.set_ylim(0, 1.0)
        ax.set_title("Effect of one reasoning-impact concept intervention")
        ax.legend()
        ax.grid(alpha=0.25)
        savefig(fig, figures / "Fig_intervention_before_after")

    # ------------------------------------------------------------------
    # Figure 5: concept performance across seeds
    # ------------------------------------------------------------------
    concept_file = tables / "concept_results_per_seed.csv"
    if concept_file.exists():
        c = pd.read_csv(concept_file)
        summary = (
            c.groupby("concept", as_index=False)
            .agg(
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_sd=("macro_f1", "std"),
                auroc_mean=("macro_ovr_auroc", "mean"),
                auroc_sd=("macro_ovr_auroc", "std"),
            )
        )
        summary.to_csv(
            tables / "concept_performance_publication.csv",
            index=False,
        )

        y = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.barh(
            y - 0.18,
            summary["macro_f1_mean"],
            0.36,
            xerr=summary["macro_f1_sd"],
            label="Macro-F1",
            capsize=3,
        )
        ax.barh(
            y + 0.18,
            summary["auroc_mean"],
            0.36,
            xerr=summary["auroc_sd"],
            label="Macro OVR AUROC",
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            summary["concept"].str.replace("_", " ")
        )
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Performance")
        ax.set_title("Concept perception performance across seeds")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        savefig(fig, figures / "Fig_concept_performance")

    print("Publication tables:", tables)
    print("Publication figures:", figures)
    print("\nSelective bootstrap summary")
    print(bootstrap_df.to_string(index=False))


if __name__ == "__main__":
    main()
