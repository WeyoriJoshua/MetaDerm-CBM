from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def summarize_seed(seed: int, predictions_csv: Path):
    d = pd.read_csv(predictions_csv)

    required = {"label", "blackbox_prob", "action"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(
            f"{predictions_csv} missing required columns: {sorted(missing)}"
        )

    y = d["label"].astype(int).to_numpy()
    p = (d["blackbox_prob"].astype(float).to_numpy() >= 0.5).astype(int)
    action = d["action"].astype(str).str.lower().to_numpy()

    predict_mask = action == "predict"
    query_mask = action == "query"
    abstain_mask = action == "abstain"
    deferred_mask = ~predict_mask

    n = len(d)
    n_predict = int(predict_mask.sum())
    n_query = int(query_mask.sum())
    n_abstain = int(abstain_mask.sum())

    if n_predict:
        yp = y[predict_mask]
        pp = p[predict_mask]
        tn, fp, fn, tp = confusion_matrix(
            yp, pp, labels=[0, 1]
        ).ravel()

        accepted_accuracy = float(np.mean(pp == yp))
        selective_risk = float(np.mean(pp != yp))
        accepted_balanced_accuracy = float(
            balanced_accuracy_score(yp, pp)
        )
        accepted_sensitivity = safe_div(tp, tp + fn)
        accepted_specificity = safe_div(tn, tn + fp)
        accepted_precision = safe_div(tp, tp + fp)
        accepted_f1 = (
            safe_div(2 * accepted_precision * accepted_sensitivity,
                     accepted_precision + accepted_sensitivity)
            if np.isfinite(accepted_precision)
            and np.isfinite(accepted_sensitivity)
            and (accepted_precision + accepted_sensitivity) > 0
            else float("nan")
        )
    else:
        tn = fp = fn = tp = 0
        accepted_accuracy = selective_risk = float("nan")
        accepted_balanced_accuracy = float("nan")
        accepted_sensitivity = float("nan")
        accepted_specificity = float("nan")
        accepted_precision = float("nan")
        accepted_f1 = float("nan")

    benign_total = int(np.sum(y == 0))
    melanoma_total = int(np.sum(y == 1))
    benign_predict = int(np.sum(predict_mask & (y == 0)))
    melanoma_predict = int(np.sum(predict_mask & (y == 1)))

    all_errors = p != y
    n_errors = int(all_errors.sum())
    deferred_errors = int(np.sum(all_errors & deferred_mask))
    predict_errors = int(np.sum(all_errors & predict_mask))

    return {
        "seed": seed,
        "n_test": int(n),
        "n_predict": n_predict,
        "n_query": n_query,
        "n_abstain": n_abstain,
        "predict_coverage": safe_div(n_predict, n),
        "query_rate": safe_div(n_query, n),
        "abstain_rate": safe_div(n_abstain, n),
        "selective_risk": selective_risk,
        "accepted_accuracy": accepted_accuracy,
        "accepted_balanced_accuracy": accepted_balanced_accuracy,
        "accepted_sensitivity": accepted_sensitivity,
        "accepted_specificity": accepted_specificity,
        "accepted_precision": accepted_precision,
        "accepted_f1": accepted_f1,
        "accepted_tn": int(tn),
        "accepted_fp": int(fp),
        "accepted_fn": int(fn),
        "accepted_tp": int(tp),
        "benign_predict_coverage": safe_div(
            benign_predict, benign_total
        ),
        "melanoma_predict_coverage": safe_div(
            melanoma_predict, melanoma_total
        ),
        "blackbox_total_errors": n_errors,
        "errors_in_predict": predict_errors,
        "errors_deferred": deferred_errors,
        "error_capture_rate": safe_div(
            deferred_errors, n_errors
        ),
    }


def t_ci(values, confidence=0.95):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float(x[0])

    mean = float(x.mean())
    sem = float(stats.sem(x))
    h = float(
        stats.t.ppf((1 + confidence) / 2.0, n - 1) * sem
    )
    return mean - h, mean + h


def aggregate(df: pd.DataFrame):
    exclude = {
        "seed",
        "n_test",
        "n_predict",
        "n_query",
        "n_abstain",
        "accepted_tn",
        "accepted_fp",
        "accepted_fn",
        "accepted_tp",
        "blackbox_total_errors",
        "errors_in_predict",
        "errors_deferred",
    }

    rows = []
    numeric_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        vals = df[col].astype(float).to_numpy()
        finite = vals[np.isfinite(vals)]
        lo, hi = t_ci(finite)

        rows.append({
            "metric": col,
            "n_seeds": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
        })

    return pd.DataFrame(rows)


def pooled_summary(df: pd.DataFrame):
    total_n = int(df["n_test"].sum())
    total_predict = int(df["n_predict"].sum())
    total_query = int(df["n_query"].sum())
    total_abstain = int(df["n_abstain"].sum())

    tn = int(df["accepted_tn"].sum())
    fp = int(df["accepted_fp"].sum())
    fn = int(df["accepted_fn"].sum())
    tp = int(df["accepted_tp"].sum())

    total_errors = int(df["blackbox_total_errors"].sum())
    deferred_errors = int(df["errors_deferred"].sum())

    accepted_n = tn + fp + fn + tp

    precision = safe_div(tp, tp + fp)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "pooled_n_test": total_n,
        "pooled_predict_coverage": safe_div(total_predict, total_n),
        "pooled_query_rate": safe_div(total_query, total_n),
        "pooled_abstain_rate": safe_div(total_abstain, total_n),
        "pooled_selective_risk": safe_div(fp + fn, accepted_n),
        "pooled_accepted_accuracy": safe_div(tn + tp, accepted_n),
        "pooled_accepted_sensitivity": sensitivity,
        "pooled_accepted_specificity": specificity,
        "pooled_accepted_precision": precision,
        "pooled_error_capture_rate": safe_div(
            deferred_errors, total_errors
        ),
        "pooled_accepted_tn": tn,
        "pooled_accepted_fp": fp,
        "pooled_accepted_fn": fn,
        "pooled_accepted_tp": tp,
    }


def main():
    p = argparse.ArgumentParser(
        description="Cross-seed selective performance analysis"
    )
    p.add_argument(
        "--results-root",
        default="results/efficientnet_b3_final",
    )
    p.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        f = root / "final" / f"seed{seed}" / "test_predictions.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"Missing predictions for seed {seed}: {f}"
            )
        rows.append(summarize_seed(seed, f))

    per_seed = pd.DataFrame(rows)
    summary = aggregate(per_seed)
    pooled = pooled_summary(per_seed)

    out = root / "publication" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    per_seed.to_csv(
        out / "selective_performance_per_seed.csv",
        index=False,
    )
    summary.to_csv(
        out / "selective_performance_mean_sd_ci.csv",
        index=False,
    )
    pd.DataFrame([pooled]).to_csv(
        out / "selective_performance_pooled.csv",
        index=False,
    )

    key_metrics = [
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

    print("\nPer-seed selective performance")
    print(
        per_seed[
            ["seed"] + key_metrics
        ].to_string(index=False)
    )

    print("\nCross-seed mean ± SD with 95% t-CI")
    show = summary[
        summary["metric"].isin(key_metrics)
    ].copy()
    print(show.to_string(index=False))

    print("\nPooled descriptive summary")
    for k, v in pooled.items():
        print(f"{k}: {v}")

    print("\nsaved:", out)


if __name__ == "__main__":
    main()
