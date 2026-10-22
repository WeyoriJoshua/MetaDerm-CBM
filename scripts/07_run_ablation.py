from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from _common import parser, setup
from metaderm.config import load_config
from metaderm.data.derm7pt import make_loader
from metaderm.evaluation.metrics import binary_metrics
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.metacognition import (
    assign_actions,
    risk_coverage_curve,
    select_three_way_thresholds,
)
from metaderm.utils.io import ensure_dir, load_checkpoint, load_json, save_json


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def _selective_summary(labels, probs, reliability, tau_low, tau_high):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= 0.5).astype(int)
    actions = assign_actions(reliability, tau_low, tau_high)

    predict_mask = actions == "predict"
    query_mask = actions == "query"
    abstain_mask = actions == "abstain"

    if predict_mask.any():
        yp = labels[predict_mask]
        pp = pred[predict_mask]
        tn, fp, fn, tp = confusion_matrix(yp, pp, labels=[0, 1]).ravel()
        selective_risk = float(np.mean(pp != yp))
        accepted_accuracy = 1.0 - selective_risk
        accepted_balanced_accuracy = float(balanced_accuracy_score(yp, pp))
        accepted_sensitivity = _safe_div(tp, tp + fn)
        accepted_specificity = _safe_div(tn, tn + fp)
    else:
        tn = fp = fn = tp = 0
        selective_risk = float("nan")
        accepted_accuracy = float("nan")
        accepted_balanced_accuracy = float("nan")
        accepted_sensitivity = float("nan")
        accepted_specificity = float("nan")

    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    p0 = int(np.sum(predict_mask & (labels == 0)))
    p1 = int(np.sum(predict_mask & (labels == 1)))

    all_errors = pred != labels
    deferred = ~predict_mask
    error_capture = _safe_div(int(np.sum(all_errors & deferred)), int(np.sum(all_errors)))

    return {
        "predict_coverage": float(np.mean(predict_mask)),
        "query_rate": float(np.mean(query_mask)),
        "abstain_rate": float(np.mean(abstain_mask)),
        "selective_risk": selective_risk,
        "accepted_accuracy": accepted_accuracy,
        "accepted_balanced_accuracy": accepted_balanced_accuracy,
        "accepted_sensitivity": accepted_sensitivity,
        "accepted_specificity": accepted_specificity,
        "accepted_tn": int(tn),
        "accepted_fp": int(fp),
        "accepted_fn": int(fn),
        "accepted_tp": int(tp),
        "benign_predict_coverage": _safe_div(p0, n0),
        "melanoma_predict_coverage": _safe_div(p1, n1),
        "error_capture_rate": error_capture,
    }


def main():
    p = parser("Run clean frozen-model metacognitive ablations")
    p.add_argument("--ablation", default="all")
    p.add_argument("--concept-checkpoint", required=True)
    p.add_argument("--blackbox-checkpoint", required=True)
    p.add_argument("--calibration", default=None)
    args = p.parse_args()

    base_cfg, seed, device = setup(args)
    cal = load_json(args.calibration) if args.calibration else None

    variants = [("full", None)]
    if args.ablation == "all":
        files = sorted(Path("configs/ablation").glob("*.yaml"))
        variants += [(f.stem, f) for f in files]
    else:
        f = Path("configs/ablation") / f"{args.ablation}.yaml"
        variants += [(f.stem, f)]

    summary = []

    for name, overlay in variants:
        cfg = base_cfg if overlay is None else load_config(
            args.config,
            *(args.override or []),
            overlay,
        )
        if args.data_root:
            cfg["data"]["root"] = args.data_root

        cm = ConceptPerceptionModel(cfg).to(device)
        bm = BlackBoxDiagnosisModel(cfg).to(device)
        load_checkpoint(args.concept_checkpoint, cm, device)
        load_checkpoint(args.blackbox_checkpoint, bm, device)

        val = evaluate_components(
            cfg, cm, bm,
            make_loader(cfg, "valid", shuffle=False),
            device, cal,
        )
        vpred = (np.asarray(val["blackbox_prob"]) >= 0.5).astype(int)
        tau_low, tau_high = select_three_way_thresholds(
            val["labels"],
            vpred,
            val["reliability"],
            cfg["metacognition"]["target_accepted_risk"],
            cfg["metacognition"]["minimum_coverage"],
            cfg["metacognition"]["abstain_fraction_of_deferred"],
        )

        test = evaluate_components(
            cfg, cm, bm,
            make_loader(cfg, "test", shuffle=False),
            device, cal,
        )
        curve = risk_coverage_curve(
            test["labels"],
            (np.asarray(test["blackbox_prob"]) >= 0.5).astype(int),
            test["reliability"],
        )

        row = {
            "ablation": name,
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            **_selective_summary(
                test["labels"],
                test["blackbox_prob"],
                test["reliability"],
                tau_low,
                tau_high,
            ),
            "blackbox_auroc": binary_metrics(
                test["labels"],
                test["blackbox_prob"],
            )["auroc"],
        }
        summary.append(row)

        out = ensure_dir(
            Path(cfg["project"]["output_root"])
            / "ablation_v2"
            / f"seed{seed}"
            / name
        )
        pd.DataFrame(
            curve,
            columns=["coverage", "risk", "threshold"],
        ).to_csv(out / "risk_coverage.csv", index=False)
        save_json(row, out / "summary.json")

    outroot = ensure_dir(
        Path(base_cfg["project"]["output_root"])
        / "ablation_v2"
        / f"seed{seed}"
    )
    result = pd.DataFrame(summary)
    result.to_csv(outroot / "ablation_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
