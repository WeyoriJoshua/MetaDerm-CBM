from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import parser, setup
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader, load_metadata, load_split_indices
from metaderm.evaluation.metrics import binary_metrics
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.consistency import (
    build_profile_map,
    hard_profile_from_probs,
    lookup_profile_reliability,
)
from metaderm.reasoning.intervention import reasoning_impact
from metaderm.reasoning.metacognition import (
    assign_actions,
    metacognitive_features,
    reliability_score,
    select_three_way_thresholds,
)
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.utils.io import ensure_dir, load_checkpoint, load_json


def _clone_probs(probs):
    return {c: np.asarray(p).copy() for c, p in probs.items()}


def _recompute(cfg, base, cp, reasoner, pmap):
    reason = reasoner.predict(cp)
    profiles = hard_profile_from_probs(cp)
    pr = lookup_profile_reliability(
        profiles,
        pmap,
        cfg["metacognition"].get("unseen_profile_reliability", 0.0),
    )
    feat = metacognitive_features(
        cp,
        base["blackbox_prob"],
        reason["probability"],
        pr,
    )
    rel = reliability_score(feat, cfg)
    return reason, pr, feat, rel


def _uniform_corruption(cp, cases, rng, strength):
    touched = []
    for i in cases:
        c = str(rng.choice(CONCEPT_COLUMNS))
        k = cp[c].shape[1]
        uniform = np.full(k, 1.0 / k)
        cp[c][i] = (1.0 - strength) * cp[c][i] + strength * uniform
        touched.append((int(i), c))
    return touched


def _confident_wrong_corruption(cp, base, cases, rng):
    touched = []
    for i in cases:
        c = str(rng.choice(CONCEPT_COLUMNS))
        k = cp[c].shape[1]
        true_state = int(base["concept_raw"]["targets"][c][i])
        choices = [j for j in range(k) if j != true_state]
        wrong = int(rng.choice(choices))
        cp[c][i, :] = 0.0
        cp[c][i, wrong] = 1.0
        touched.append((int(i), c))
    return touched


def _adversarial_wrong_corruption(cp, base, cases, reasoner):
    touched = []
    for i in cases:
        base_prob = float(reasoner.predict({
            c: cp[c][i:i+1] for c in CONCEPT_COLUMNS
        })["probability"][0])

        best = None
        for c in CONCEPT_COLUMNS:
            k = cp[c].shape[1]
            true_state = int(base["concept_raw"]["targets"][c][i])
            for state in range(k):
                if state == true_state:
                    continue
                one = {name: cp[name][i:i+1].copy() for name in CONCEPT_COLUMNS}
                one[c][0, :] = 0.0
                one[c][0, state] = 1.0
                p = float(reasoner.predict(one)["probability"][0])
                score = abs(p - base_prob)
                if best is None or score > best[0]:
                    best = (score, c, state)
        _, c, state = best
        cp[c][i, :] = 0.0
        cp[c][i, int(state)] = 1.0
        touched.append((int(i), c))
    return touched


def main():
    p = parser("Harder concept-corruption robustness experiment")
    p.add_argument("--concept-checkpoint", required=True)
    p.add_argument("--blackbox-checkpoint", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--rates", default="0.05,0.10,0.20,0.30")
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    cfg, seed, device = setup(args)
    cal = load_json(args.calibration)

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
        val["labels"], vpred, val["reliability"],
        cfg["metacognition"]["target_accepted_risk"],
        cfg["metacognition"]["minimum_coverage"],
        cfg["metacognition"]["abstain_fraction_of_deferred"],
    )

    base = evaluate_components(
        cfg, cm, bm,
        make_loader(cfg, "test", shuffle=False),
        device, cal,
    )
    reasoner = SoftSevenPointReasoner(cfg)
    meta = load_metadata(cfg)
    pmap = build_profile_map(
        meta.iloc[load_split_indices(cfg, "train")],
        cfg,
    )

    baseline_reason = binary_metrics(
        base["labels"],
        base["reason"]["probability"],
    )
    baseline_rel = float(np.mean(base["reliability"]))

    rows = []
    n = len(base["labels"])

    for mode in ["uniform", "confident_wrong", "adversarial_wrong"]:
        for rate in [float(x) for x in args.rates.split(",")]:
            for repeat in range(args.repeats):
                rng = np.random.default_rng(seed + repeat + int(rate * 10000))
                m = max(1, int(round(rate * n)))
                cases = rng.choice(n, size=m, replace=False)
                corrupted = np.zeros(n, dtype=int)
                corrupted[cases] = 1

                cp = _clone_probs(base["concept_probs"])

                if mode == "uniform":
                    _uniform_corruption(cp, cases, rng, args.strength)
                elif mode == "confident_wrong":
                    _confident_wrong_corruption(cp, base, cases, rng)
                else:
                    _adversarial_wrong_corruption(cp, base, cases, reasoner)

                reason, pr, feat, rel = _recompute(
                    cfg, base, cp, reasoner, pmap
                )
                met = binary_metrics(
                    base["labels"],
                    reason["probability"],
                )
                detect_auc = (
                    roc_auc_score(corrupted, 1.0 - rel)
                    if len(np.unique(corrupted)) == 2
                    else float("nan")
                )

                actions = assign_actions(rel, tau_low, tau_high)
                pred = (np.asarray(base["blackbox_prob"]) >= 0.5).astype(int)
                pmask = actions == "predict"
                risk = (
                    float(np.mean(pred[pmask] != base["labels"][pmask]))
                    if pmask.any()
                    else float("nan")
                )

                rows.append({
                    "mode": mode,
                    "corruption_rate": rate,
                    "repeat": repeat,
                    "strength": args.strength,
                    "reason_auroc": met["auroc"],
                    "reason_f1": met["f1"],
                    "delta_reason_auroc": met["auroc"] - baseline_reason["auroc"],
                    "delta_reason_f1": met["f1"] - baseline_reason["f1"],
                    "mean_reliability": float(np.mean(rel)),
                    "delta_mean_reliability": float(np.mean(rel)) - baseline_rel,
                    "mean_reliability_corrupted": float(np.mean(rel[corrupted == 1])),
                    "mean_reliability_clean": float(np.mean(rel[corrupted == 0])),
                    "corruption_detection_auroc": float(detect_auc),
                    "predict_coverage": float(np.mean(pmask)),
                    "selective_risk": risk,
                })

    out = ensure_dir(
        Path(cfg["project"]["output_root"])
        / "robustness_v2"
        / f"seed{seed}"
    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness_repeats.csv", index=False)

    summary = (
        df.groupby(["mode", "corruption_rate"], as_index=False)
        .agg(
            reason_auroc_mean=("reason_auroc", "mean"),
            delta_reason_auroc_mean=("delta_reason_auroc", "mean"),
            reason_f1_mean=("reason_f1", "mean"),
            delta_reason_f1_mean=("delta_reason_f1", "mean"),
            mean_reliability_mean=("mean_reliability", "mean"),
            delta_mean_reliability_mean=("delta_mean_reliability", "mean"),
            corruption_detection_auroc_mean=("corruption_detection_auroc", "mean"),
            predict_coverage_mean=("predict_coverage", "mean"),
            selective_risk_mean=("selective_risk", "mean"),
        )
    )
    summary.to_csv(out / "robustness_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
