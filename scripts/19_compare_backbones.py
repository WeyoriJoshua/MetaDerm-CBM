from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_diag(root: Path, label: str):
    f = root / "publication" / "tables" / "diagnostic_results_per_seed.csv"
    d = pd.read_csv(f)
    d["backbone"] = label
    return d


def load_sel(root: Path, label: str):
    f = root / "publication" / "tables" / "selective_performance_per_seed.csv"
    d = pd.read_csv(f)
    d["backbone"] = label
    return d


def paired_summary(df, metric, model=None):
    x = df.copy()
    if model is not None:
        x = x[x["model"] == model]
    pivot = x.pivot(index="seed", columns="backbone", values=metric)
    if not {"EfficientNet-B3", "Swin-Tiny"}.issubset(pivot.columns):
        return None
    delta = pivot["Swin-Tiny"] - pivot["EfficientNet-B3"]
    return {
        "metric": metric,
        "model": model or "selective_gate",
        "efficientnet_mean": float(pivot["EfficientNet-B3"].mean()),
        "swin_mean": float(pivot["Swin-Tiny"].mean()),
        "swin_minus_efficientnet_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--efficientnet-root",
        default="results/efficientnet_b3_final",
    )
    p.add_argument(
        "--swin-root",
        default="results/swin_tiny_final",
    )
    args = p.parse_args()

    eroot = Path(args.efficientnet_root)
    sroot = Path(args.swin_root)

    diag = pd.concat(
        [
            load_diag(eroot, "EfficientNet-B3"),
            load_diag(sroot, "Swin-Tiny"),
        ],
        ignore_index=True,
    )
    sel = pd.concat(
        [
            load_sel(eroot, "EfficientNet-B3"),
            load_sel(sroot, "Swin-Tiny"),
        ],
        ignore_index=True,
    )

    rows = []
    for model in ["blackbox", "soft_reasoner"]:
        for metric in [
            "auroc",
            "balanced_accuracy",
            "sensitivity_recall",
            "specificity",
            "f1",
            "auprc",
        ]:
            r = paired_summary(diag, metric, model=model)
            if r:
                rows.append(r)

    for metric in [
        "predict_coverage",
        "selective_risk",
        "accepted_balanced_accuracy",
        "accepted_sensitivity",
        "accepted_specificity",
        "melanoma_predict_coverage",
        "error_capture_rate",
    ]:
        r = paired_summary(sel, metric, model=None)
        if r:
            rows.append(r)

    out = sroot / "publication" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(
        out / "backbone_sensitivity_comparison.csv",
        index=False,
    )
    print(result.to_string(index=False))
    print("\nsaved:", out / "backbone_sensitivity_comparison.csv")


if __name__ == "__main__":
    main()
