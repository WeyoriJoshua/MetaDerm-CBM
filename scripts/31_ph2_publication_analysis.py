
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


DIAG_METRICS = [
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

SELECTIVE_METRICS = [
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


def percentile_ci(values, rng, n_boot=10000, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize_by_group(df, group_cols, metrics, rng, n_boot):
    rows = []
    if not isinstance(group_cols, list):
        group_cols = [group_cols]

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = dict(zip(group_cols, keys))

        for metric in metrics:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            if len(x) == 0:
                continue
            lo, hi = percentile_ci(x, rng, n_boot=n_boot)
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "n_seeds": int(len(x)),
                    "mean": float(np.mean(x)),
                    "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                    "median": float(np.median(x)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def find_internal_csv(root: Path, required_cols, preferred_names=()):
    for name in preferred_names:
        hits = list(root.rglob(name))
        for path in hits:
            try:
                df = pd.read_csv(path)
                if set(required_cols).issubset(df.columns):
                    return path, df
            except Exception:
                pass

    for path in root.rglob("*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if set(required_cols).issubset(df.columns):
            return path, df

    raise FileNotFoundError(
        f"Could not find CSV under {root} with columns {required_cols}"
    )


def binary_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def paired_bootstrap_seed(pred_df, n_boot, rng):
    y = pred_df["label"].to_numpy(dtype=int)
    pb = pred_df["blackbox_prob"].to_numpy(dtype=float)
    pr = pred_df["reason_prob"].to_numpy(dtype=float)

    n = len(y)
    base_b = binary_metrics(y, pb)
    base_r = binary_metrics(y, pr)

    metrics = list(base_b.keys())
    diffs = {m: [] for m in metrics}

    attempts = 0
    while len(diffs["auroc"]) < n_boot and attempts < n_boot * 3:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yy = y[idx]

        if len(np.unique(yy)) < 2:
            continue

        bb = binary_metrics(yy, pb[idx])
        rr = binary_metrics(yy, pr[idx])

        for m in metrics:
            diffs[m].append(rr[m] - bb[m])

    rows = []
    for m in metrics:
        arr = np.asarray(diffs[m], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            lo = hi = p_two = np.nan
        else:
            lo, hi = np.quantile(arr, [0.025, 0.975])
            # Descriptive bootstrap tail probability, not a replacement for
            # a pre-specified inferential test.
            p_two = min(
                1.0,
                2.0 * min(
                    np.mean(arr <= 0),
                    np.mean(arr >= 0),
                ),
            )

        rows.append(
            {
                "metric": m,
                "blackbox": base_b[m],
                "soft_reasoner": base_r[m],
                "reasoner_minus_blackbox": base_r[m] - base_b[m],
                "paired_bootstrap_ci_low": float(lo),
                "paired_bootstrap_ci_high": float(hi),
                "paired_bootstrap_tail_p": float(p_two),
                "bootstrap_replicates": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def mcnemar_exact(pred_df):
    y = pred_df["label"].to_numpy(dtype=int)
    b = (pred_df["blackbox_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
    r = (pred_df["reason_prob"].to_numpy(dtype=float) >= 0.5).astype(int)

    b_ok = b == y
    r_ok = r == y

    b_correct_r_wrong = int(np.sum(b_ok & (~r_ok)))
    b_wrong_r_correct = int(np.sum((~b_ok) & r_ok))
    discordant = b_correct_r_wrong + b_wrong_r_correct

    if discordant == 0:
        p = 1.0
    elif binomtest is not None:
        p = float(
            binomtest(
                min(b_correct_r_wrong, b_wrong_r_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        # Exact two-sided binomial fallback.
        from math import comb
        k = min(b_correct_r_wrong, b_wrong_r_correct)
        tail = sum(comb(discordant, i) for i in range(0, k + 1))
        p = min(1.0, 2.0 * tail / (2 ** discordant))

    return {
        "blackbox_correct_reasoner_wrong": b_correct_r_wrong,
        "blackbox_wrong_reasoner_correct": b_wrong_r_correct,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p,
    }


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)

    finite = np.flatnonzero(np.isfinite(pvals))
    if len(finite) == 0:
        return out

    order_local = np.argsort(pvals[finite])
    ordered_idx = finite[order_local]
    m = len(ordered_idx)

    running = 0.0
    for rank, idx in enumerate(ordered_idx):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)

    return out


def seed_delta_summary(external_diag, rng, n_boot):
    pivot = external_diag.pivot(index="seed", columns="model")
    rows = []

    for metric in DIAG_METRICS:
        if metric not in external_diag.columns:
            continue
        if (metric, "blackbox") not in pivot.columns:
            continue
        if (metric, "soft_reasoner") not in pivot.columns:
            continue

        bb = pivot[(metric, "blackbox")].astype(float)
        rr = pivot[(metric, "soft_reasoner")].astype(float)
        delta = (rr - bb).dropna()

        lo, hi = percentile_ci(delta.to_numpy(), rng, n_boot=n_boot)

        rows.append(
            {
                "metric": metric,
                "n_seeds": len(delta),
                "blackbox_mean": float(bb.mean()),
                "reasoner_mean": float(rr.mean()),
                "reasoner_minus_blackbox_mean": float(delta.mean()),
                "delta_sd": float(delta.std(ddof=1)),
                "seed_bootstrap_ci_low": lo,
                "seed_bootstrap_ci_high": hi,
                "delta_min": float(delta.min()),
                "delta_max": float(delta.max()),
            }
        )

    return pd.DataFrame(rows)


def internal_external_summary(internal_diag, external_diag):
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

    rows = []
    for model in ["blackbox", "soft_reasoner"]:
        a = internal_diag[internal_diag["model"] == model]
        b = external_diag[external_diag["model"] == model]

        common = sorted(set(a["seed"]) & set(b["seed"]))
        a = a[a["seed"].isin(common)].set_index("seed")
        b = b[b["seed"].isin(common)].set_index("seed")

        for metric in metrics:
            if metric not in a.columns or metric not in b.columns:
                continue
            av = pd.to_numeric(a[metric], errors="coerce")
            bv = pd.to_numeric(b[metric], errors="coerce")
            d = bv - av
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_seeds": len(common),
                    "internal_mean": float(av.mean()),
                    "internal_sd": float(av.std(ddof=1)),
                    "external_mean": float(bv.mean()),
                    "external_sd": float(bv.std(ddof=1)),
                    "external_minus_internal_mean": float(d.mean()),
                    "delta_sd": float(d.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def save_fig(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_external_diagnostic(summary, outdir):
    metrics = [
        "auroc",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "f1",
        "auprc",
    ]
    labels = [
        "AUROC",
        "Balanced accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "AUPRC",
    ]

    models = ["blackbox", "soft_reasoner"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for i, model in enumerate(models):
        g = summary[
            (summary["model"] == model)
            & (summary["metric"].isin(metrics))
        ].set_index("metric")

        means = [g.loc[m, "mean"] for m in metrics]
        sds = [g.loc[m, "sd"] for m in metrics]

        pos = x + (i - 0.5) * width
        ax.bar(
            pos,
            means,
            width,
            yerr=sds,
            capsize=3,
            label=("Black-box" if model == "blackbox" else "Soft reasoner"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PH² external diagnostic performance across five frozen seeds")
    ax.legend()

    save_fig(fig, outdir / "Fig_PH2_external_diagnostic")


def figure_internal_external(comp, outdir):
    metrics = [
        "auroc",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
    ]
    labels = [
        "AUROC",
        "Balanced accuracy",
        "Sensitivity",
        "Specificity",
    ]

    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    series = [
        ("blackbox", "internal_mean", "Black-box: Derm7pt"),
        ("blackbox", "external_mean", "Black-box: PH²"),
        ("soft_reasoner", "internal_mean", "Reasoner: Derm7pt"),
        ("soft_reasoner", "external_mean", "Reasoner: PH²"),
    ]

    for i, (model, col, label) in enumerate(series):
        g = comp[
            (comp["model"] == model)
            & (comp["metric"].isin(metrics))
        ].set_index("metric")

        means = [g.loc[m, col] for m in metrics]
        sd_col = "internal_sd" if col == "internal_mean" else "external_sd"
        sds = [g.loc[m, sd_col] for m in metrics]

        pos = x + (i - 1.5) * width
        ax.bar(
            pos,
            means,
            width,
            yerr=sds,
            capsize=3,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Internal Derm7pt versus external PH² performance")
    ax.legend(ncol=2)

    save_fig(fig, outdir / "Fig_internal_external_comparison")


def figure_selective(summary, outdir):
    metrics = [
        "predict_coverage",
        "query_rate",
        "abstain_rate",
        "error_capture_rate",
        "benign_predict_coverage",
        "melanoma_predict_coverage",
    ]
    labels = [
        "PREDICT\ncoverage",
        "QUERY\nrate",
        "ABSTAIN\nrate",
        "Error\ncapture",
        "Benign\nPREDICT",
        "Melanoma\nPREDICT",
    ]

    g = summary.set_index("metric")
    means = [g.loc[m, "mean"] for m in metrics]
    sds = [g.loc[m, "sd"] for m in metrics]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=sds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title("PH² external selective-decision behaviour")

    save_fig(fig, outdir / "Fig_PH2_selective_behaviour")


def figure_concepts(summary, outdir):
    ordered = [
        "pigment_network_typical_vs_atypical",
        "dots_and_globules_3state",
        "streaks_presence",
        "regression_structures_presence",
        "blue_whitish_veil_presence",
    ]
    labels = [
        "Pigment\nnetwork",
        "Dots /\nglobules",
        "Streaks",
        "Regression\nareas",
        "Blue-whitish\nveil",
    ]

    g = summary.set_index("concept")
    means = [g.loc[c, "auroc_mean"] for c in ordered]
    sds = [g.loc[c, "auroc_sd"] for c in ordered]

    fig, ax = plt.subplots(figsize=(9.3, 5.3))
    x = np.arange(len(ordered))
    ax.bar(x, means, yerr=sds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("PH² external semantic concept transfer")

    save_fig(fig, outdir / "Fig_PH2_concept_transfer")


def write_results_insert(
    outpath,
    ext_summary,
    selective_summary,
    concept_summary,
):
    d = ext_summary.set_index(["model", "metric"])
    s = selective_summary.set_index("metric")
    c = concept_summary.set_index("concept")

    def ms(model, metric):
        row = d.loc[(model, metric)]
        return f"{row['mean']:.3f} ± {row['sd']:.3f}"

    text = f"""# PH² External Validation — Manuscript-Ready Results Insert

The frozen MetaDerm-CBM configuration was evaluated on the independent PH² dataset (n=200; 40 melanomas and 160 non-melanomas) across the five training seeds without retraining, recalibration, threshold optimisation, or modification of the metacognitive policy. Across seeds, the direct black-box classifier achieved AUROC {ms('blackbox','auroc')}, balanced accuracy {ms('blackbox','balanced_accuracy')}, and melanoma sensitivity {ms('blackbox','sensitivity_recall')}. The soft concept-based reasoner achieved AUROC {ms('soft_reasoner','auroc')}, balanced accuracy {ms('soft_reasoner','balanced_accuracy')}, and melanoma sensitivity {ms('soft_reasoner','sensitivity_recall')}. External reasoner sensitivity was therefore broadly preserved relative to the internal Derm7pt evaluation, while discrimination, specificity, F1, AUPRC, Brier score, and calibration error were improved on PH². These differences should be interpreted as dataset-specific external performance rather than evidence that unseen data are generally easier or that the model will necessarily improve under arbitrary domain shift.

The frozen metacognitive gate retained a selective triage pattern on PH². Mean PREDICT coverage was {s.loc['predict_coverage','mean']:.3f} ± {s.loc['predict_coverage','sd']:.3f}, QUERY rate was {s.loc['query_rate','mean']:.3f} ± {s.loc['query_rate','sd']:.3f}, and ABSTAIN rate was {s.loc['abstain_rate','mean']:.3f} ± {s.loc['abstain_rate','sd']:.3f}. The mean selective risk among PREDICT cases was {s.loc['selective_risk','mean']:.3f} ± {s.loc['selective_risk','sd']:.3f}, while {s.loc['error_capture_rate','mean']:.3f} ± {s.loc['error_capture_rate','sd']:.3f} of black-box errors were deferred to QUERY or ABSTAIN. Class-conditional coverage remained asymmetric: benign PREDICT coverage was {s.loc['benign_predict_coverage','mean']:.3f} ± {s.loc['benign_predict_coverage','sd']:.3f}, compared with melanoma PREDICT coverage of {s.loc['melanoma_predict_coverage','mean']:.3f} ± {s.loc['melanoma_predict_coverage','sd']:.3f}. Accordingly, the selective mechanism is interpreted as risk-aware decision support rather than autonomous melanoma exclusion.

Partial semantic transfer was evaluated only for PH² criteria with defensible correspondence to Derm7pt. Mean AUROC across seeds was {c.loc['pigment_network_typical_vs_atypical','auroc_mean']:.3f} for pigment-network atypia, {c.loc['dots_and_globules_3state','auroc_mean']:.3f} for dots/globules, {c.loc['streaks_presence','auroc_mean']:.3f} for streak presence, {c.loc['regression_structures_presence','auroc_mean']:.3f} for regression-area presence, and {c.loc['blue_whitish_veil_presence','auroc_mean']:.3f} for blue-whitish veil. The uneven concept-transfer performance indicates that external diagnostic reasoning can remain useful despite imperfect transfer of individual semantic representations.
"""
    outpath.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Publication analysis for frozen PH2 external validation."
    )
    parser.add_argument(
        "--external-root",
        default="results/ph2_external/efficientnet_b3",
    )
    parser.add_argument(
        "--internal-root",
        default="results/efficientnet_b3_final",
    )
    parser.add_argument(
        "--output-root",
        default="results/ph2_external/efficientnet_b3/publication",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    eroot = Path(args.external_root)
    iroot = Path(args.internal_root)
    outroot = Path(args.output_root)
    tables = outroot / "tables"
    figures = outroot / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    ext_diag = pd.read_csv(eroot / "diagnostic_per_seed.csv")
    ext_sel = pd.read_csv(eroot / "selective_per_seed.csv")
    ext_concepts = pd.read_csv(eroot / "concept_transfer_per_seed.csv")

    internal_diag_path, internal_diag = find_internal_csv(
        iroot,
        required_cols=["seed", "model", "auroc", "balanced_accuracy"],
        preferred_names=("diagnostic_results_per_seed.csv",),
    )
    print("internal diagnostic source:", internal_diag_path)

    try:
        internal_sel_path, internal_sel = find_internal_csv(
            iroot,
            required_cols=["seed", "predict_coverage", "error_capture_rate"],
            preferred_names=(
                "selective_performance_per_seed.csv",
                "selective_results_per_seed.csv",
                "selective_per_seed.csv",
            ),
        )
        print("internal selective source:", internal_sel_path)
    except Exception as exc:
        internal_sel = None
        print("WARNING: internal selective CSV not found:", exc)

    ext_diag_summary = summarize_by_group(
        ext_diag,
        ["model"],
        DIAG_METRICS,
        rng,
        args.bootstrap_iterations,
    )
    ext_diag_summary.to_csv(
        tables / "Table_PH2_external_diagnostic_summary.csv",
        index=False,
    )

    ext_sel_summary = summarize_by_group(
        ext_sel.assign(group="PH2"),
        ["group"],
        SELECTIVE_METRICS,
        rng,
        args.bootstrap_iterations,
    ).drop(columns=["group"])
    ext_sel_summary.to_csv(
        tables / "Table_PH2_selective_summary.csv",
        index=False,
    )

    concept_summary = (
        ext_concepts.groupby("concept", as_index=False)
        .agg(
            n=("n", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
        )
    )

    concept_ci_rows = []
    for concept, g in ext_concepts.groupby("concept"):
        for metric in ["accuracy", "macro_f1", "auroc"]:
            lo, hi = percentile_ci(
                g[metric].to_numpy(),
                rng,
                n_boot=args.bootstrap_iterations,
            )
            concept_ci_rows.append(
                {
                    "concept": concept,
                    "metric": metric,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )
    concept_ci = pd.DataFrame(concept_ci_rows)

    concept_summary.to_csv(
        tables / "Table_PH2_concept_transfer_summary.csv",
        index=False,
    )
    concept_ci.to_csv(
        tables / "Table_PH2_concept_transfer_seed_bootstrap_ci.csv",
        index=False,
    )

    seed_deltas = seed_delta_summary(
        ext_diag,
        rng,
        args.bootstrap_iterations,
    )
    seed_deltas.to_csv(
        tables / "Table_PH2_reasoner_vs_blackbox_seed_deltas.csv",
        index=False,
    )

    paired_frames = []
    mcnemar_rows = []

    for seed_dir in sorted(eroot.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
            continue

        pred_path = seed_dir / "ph2_predictions.csv"
        if not pred_path.exists():
            continue

        pred = pd.read_csv(pred_path)

        paired = paired_bootstrap_seed(
            pred,
            args.bootstrap_iterations,
            np.random.default_rng(args.seed + seed),
        )
        paired.insert(0, "seed", seed)
        paired_frames.append(paired)

        mc = mcnemar_exact(pred)
        mcnemar_rows.append({"seed": seed, **mc})

    paired_all = pd.concat(paired_frames, ignore_index=True)
    paired_all.to_csv(
        tables / "Table_PH2_paired_bootstrap_per_seed.csv",
        index=False,
    )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    if len(mcnemar_df):
        mcnemar_df["holm_adjusted_p"] = holm_adjust(
            mcnemar_df["mcnemar_exact_p"].to_numpy()
        )
    mcnemar_df.to_csv(
        tables / "Table_PH2_McNemar_per_seed.csv",
        index=False,
    )

    int_ext = internal_external_summary(
        internal_diag,
        ext_diag,
    )
    int_ext.to_csv(
        tables / "Table_internal_external_comparison.csv",
        index=False,
    )

    if internal_sel is not None:
        common = sorted(
            set(internal_sel["seed"]) & set(ext_sel["seed"])
        )
        rows = []
        for metric in SELECTIVE_METRICS:
            if metric not in internal_sel.columns or metric not in ext_sel.columns:
                continue
            a = (
                internal_sel[internal_sel["seed"].isin(common)]
                .set_index("seed")[metric]
                .astype(float)
            )
            b = (
                ext_sel[ext_sel["seed"].isin(common)]
                .set_index("seed")[metric]
                .astype(float)
            )
            d = b - a
            rows.append(
                {
                    "metric": metric,
                    "internal_mean": a.mean(),
                    "internal_sd": a.std(ddof=1),
                    "external_mean": b.mean(),
                    "external_sd": b.std(ddof=1),
                    "external_minus_internal_mean": d.mean(),
                    "delta_sd": d.std(ddof=1),
                }
            )
        pd.DataFrame(rows).to_csv(
            tables / "Table_selective_internal_external_comparison.csv",
            index=False,
        )

    figure_external_diagnostic(
        ext_diag_summary,
        figures,
    )
    figure_internal_external(
        int_ext,
        figures,
    )
    figure_selective(
        ext_sel_summary,
        figures,
    )
    figure_concepts(
        concept_summary,
        figures,
    )

    write_results_insert(
        outroot / "PH2_RESULTS_INSERT.md",
        ext_diag_summary,
        ext_sel_summary,
        concept_summary,
    )

    manifest = {
        "external_root": str(eroot),
        "internal_root": str(iroot),
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.seed,
        "analysis_note": (
            "PH2 remained locked. No model, calibration, threshold, "
            "metacognitive-weight, reasoning-rule, or query-policy tuning "
            "was performed in this analysis."
        ),
    }
    (outroot / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== PH2 PUBLICATION ANALYSIS COMPLETE ===")
    print("Tables:")
    for p in sorted(tables.glob("*.csv")):
        print(" ", p)
    print("Figures:")
    for p in sorted(figures.glob("*.png")):
        print(" ", p)
    print("Manuscript insert:")
    print(" ", outroot / "PH2_RESULTS_INSERT.md")


if __name__ == "__main__":
    main()
