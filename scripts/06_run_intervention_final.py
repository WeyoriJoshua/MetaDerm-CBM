from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd

from _common import parser, setup
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader
from metaderm.evaluation.metrics import binary_metrics
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.consistency import (
    hard_profile_from_probs,
    lookup_profile_reliability,
)
from metaderm.reasoning.intervention import (
    apply_oracle_correction,
    reasoning_impact,
)
from metaderm.reasoning.metacognition import (
    assign_actions,
    metacognitive_features,
    reliability_score,
    select_three_way_thresholds,
)
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.utils.io import (
    ensure_dir,
    load_checkpoint,
    load_json,
    save_json,
)


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _select_impact_only(reasoner, concept_probs, sample_index: int):
    ranked = []
    for concept in CONCEPT_COLUMNS:
        impact = reasoning_impact(
            reasoner,
            concept_probs,
            sample_index,
            concept,
        )
        ranked.append((float(impact), concept))
    ranked.sort(reverse=True)
    return ranked[0], ranked


def _single_sample_post_intervention(
    cfg,
    reasoner,
    test_eval,
    sample_index: int,
    corrected_probs,
    tau_low: float,
    tau_high: float,
):
    reason_after = reasoner.predict(corrected_probs)
    reason_prob_after = _scalar(reason_after["probability"])

    corrected_profile = hard_profile_from_probs(corrected_probs)
    profile_reliability_after = _scalar(
        lookup_profile_reliability(
            corrected_profile,
            test_eval["profile_map"],
            cfg["metacognition"].get(
                "unseen_profile_reliability",
                0.0,
            ),
        )
    )

    blackbox_prob = np.asarray(
        [_scalar(test_eval["blackbox_prob"][sample_index])]
    )
    reason_prob = np.asarray([reason_prob_after])
    profile_rel = np.asarray([profile_reliability_after])

    features_after = metacognitive_features(
        corrected_probs,
        blackbox_prob,
        reason_prob,
        profile_rel,
    )
    reliability_after = _scalar(
        reliability_score(features_after, cfg)
    )
    action_after = str(
        assign_actions(
            np.asarray([reliability_after]),
            tau_low,
            tau_high,
        )[0]
    )

    return {
        "reason_prob_after": reason_prob_after,
        "profile_reliability_after": profile_reliability_after,
        "reliability_after": reliability_after,
        "action_after": action_after,
        "concept_uncertainty_after": _scalar(
            features_after["concept_uncertainty"]
        ),
        "diagnosis_uncertainty_after": _scalar(
            features_after["diagnosis_uncertainty"]
        ),
        "neural_symbolic_disagreement_after": _scalar(
            features_after["neural_symbolic_disagreement"]
        ),
    }


def main():
    p = parser(
        "Final QUERY-only impact-first expert concept intervention"
    )
    p.add_argument("--concept-checkpoint", required=True)
    p.add_argument("--blackbox-checkpoint", required=True)
    p.add_argument("--calibration", required=True)
    args = p.parse_args()

    cfg, seed, device = setup(args)
    calibration = load_json(args.calibration)

    concept_model = ConceptPerceptionModel(cfg).to(device)
    blackbox_model = BlackBoxDiagnosisModel(cfg).to(device)
    load_checkpoint(
        args.concept_checkpoint,
        concept_model,
        device,
    )
    load_checkpoint(
        args.blackbox_checkpoint,
        blackbox_model,
        device,
    )

    # Thresholds are frozen from validation only.
    val = evaluate_components(
        cfg,
        concept_model,
        blackbox_model,
        make_loader(cfg, "valid", shuffle=False),
        device,
        calibration,
    )
    val_pred = (
        np.asarray(val["blackbox_prob"]) >= 0.5
    ).astype(int)

    tau_low, tau_high = select_three_way_thresholds(
        val["labels"],
        val_pred,
        val["reliability"],
        cfg["metacognition"]["target_accepted_risk"],
        cfg["metacognition"]["minimum_coverage"],
        cfg["metacognition"]["abstain_fraction_of_deferred"],
    )

    test = evaluate_components(
        cfg,
        concept_model,
        blackbox_model,
        make_loader(cfg, "test", shuffle=False),
        device,
        calibration,
    )

    actions_before = assign_actions(
        test["reliability"],
        tau_low,
        tau_high,
    )
    query_indices = np.flatnonzero(
        actions_before == "query"
    )

    reasoner = SoftSevenPointReasoner(cfg)
    rows = []

    for i in query_indices:
        (impact, concept), _ranking = _select_impact_only(
            reasoner,
            test["concept_probs"],
            int(i),
        )

        before_vec = np.asarray(
            test["concept_probs"][concept][i]
        )
        predicted_state = int(before_vec.argmax())
        true_state = int(
            test["concept_raw"]["targets"][concept][i]
        )
        concept_was_correct = int(
            predicted_state == true_state
        )

        corrected_probs = apply_oracle_correction(
            test["concept_probs"],
            int(i),
            concept,
            true_state,
        )

        reason_prob_before = _scalar(
            test["reason"]["probability"][i]
        )
        label = int(test["labels"][i])
        reason_pred_before = int(
            reason_prob_before >= 0.5
        )
        reason_correct_before = int(
            reason_pred_before == label
        )

        post = _single_sample_post_intervention(
            cfg,
            reasoner,
            test,
            int(i),
            corrected_probs,
            tau_low,
            tau_high,
        )

        reason_prob_after = post[
            "reason_prob_after"
        ]
        reason_pred_after = int(
            reason_prob_after >= 0.5
        )
        reason_correct_after = int(
            reason_pred_after == label
        )

        rows.append(
            {
                "source_index": int(
                    test["source_indices"][i]
                ),
                "label": label,
                "blackbox_prob": _scalar(
                    test["blackbox_prob"][i]
                ),
                "queried_concept": concept,
                "query_policy": "reasoning_impact",
                "reasoning_impact": float(impact),
                "predicted_concept_state": predicted_state,
                "true_concept_state": true_state,
                "concept_was_correct": concept_was_correct,
                "intervention_type": (
                    "verification"
                    if concept_was_correct
                    else "correction"
                ),
                "reason_prob_before": reason_prob_before,
                "reason_prob_after": reason_prob_after,
                "reason_probability_change": (
                    reason_prob_after
                    - reason_prob_before
                ),
                "reason_correct_before": reason_correct_before,
                "reason_correct_after": reason_correct_after,
                "error_recovered": int(
                    reason_correct_before == 0
                    and reason_correct_after == 1
                ),
                "error_introduced": int(
                    reason_correct_before == 1
                    and reason_correct_after == 0
                ),
                "profile_reliability_before": _scalar(
                    test["profile_reliability"][i]
                ),
                "profile_reliability_after": post[
                    "profile_reliability_after"
                ],
                "reliability_before": _scalar(
                    test["reliability"][i]
                ),
                "reliability_after": post[
                    "reliability_after"
                ],
                "reliability_change": (
                    post["reliability_after"]
                    - _scalar(test["reliability"][i])
                ),
                "action_before": "query",
                "action_after": post[
                    "action_after"
                ],
            }
        )

    out = ensure_dir(
        Path(cfg["project"]["output_root"])
        / "intervention_final"
        / f"seed{seed}"
    )

    df = pd.DataFrame(rows)
    df.to_csv(
        out / "query_only_impact_intervention.csv",
        index=False,
    )

    if len(df) == 0:
        summary = {
            "seed": seed,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "n_test": int(len(test["labels"])),
            "n_query": 0,
        }
        save_json(
            summary,
            out / "summary.json",
        )
        print(summary)
        print("saved:", out)
        return

    before_metrics = binary_metrics(
        df["label"].to_numpy(),
        df["reason_prob_before"].to_numpy(),
    )
    after_metrics = binary_metrics(
        df["label"].to_numpy(),
        df["reason_prob_after"].to_numpy(),
    )

    transitions = (
        df.groupby(
            ["action_before", "action_after"]
        )
        .size()
        .reset_index(name="n")
    )
    transitions.to_csv(
        out / "action_transitions.csv",
        index=False,
    )

    concept_counts = (
        df["queried_concept"]
        .value_counts()
        .rename_axis("queried_concept")
        .reset_index(name="n")
    )
    concept_counts["fraction"] = (
        concept_counts["n"] / len(df)
    )
    concept_counts.to_csv(
        out / "queried_concept_distribution.csv",
        index=False,
    )

    n_recovered = int(
        df["error_recovered"].sum()
    )
    n_introduced = int(
        df["error_introduced"].sum()
    )

    summary = {
        "seed": seed,
        "query_policy": "reasoning_impact",
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "n_test": int(len(test["labels"])),
        "n_query": int(len(df)),
        "query_rate": float(
            len(df) / len(test["labels"])
        ),
        "selected_concept_correction_rate": float(
            1.0 - df["concept_was_correct"].mean()
        ),
        "reasoner_before": before_metrics,
        "reasoner_after": after_metrics,
        "absolute_reason_accuracy_gain": float(
            after_metrics["accuracy"]
            - before_metrics["accuracy"]
        ),
        "error_recovery_count": n_recovered,
        "error_introduced_count": n_introduced,
        "net_error_reduction": int(
            n_recovered - n_introduced
        ),
        "mean_reliability_before": float(
            df["reliability_before"].mean()
        ),
        "mean_reliability_after": float(
            df["reliability_after"].mean()
        ),
        "post_intervention_action_rates": {
            str(k): float(v)
            for k, v in (
                df["action_after"]
                .value_counts(normalize=True)
                .to_dict()
                .items()
            )
        },
    }

    save_json(
        summary,
        out / "summary.json",
    )

    print("FINAL QUERY-only impact-first intervention")
    print("n_query:", summary["n_query"])
    print(
        "selected concept required correction:",
        f"{100.0 * summary['selected_concept_correction_rate']:.2f}%",
    )
    print(
        "reasoner accuracy:",
        f"{before_metrics['accuracy']:.4f} -> "
        f"{after_metrics['accuracy']:.4f}",
    )
    print(
        "error recovery:",
        n_recovered,
        "introduced:",
        n_introduced,
        "net:",
        n_recovered - n_introduced,
    )
    print(
        "mean reliability:",
        f"{summary['mean_reliability_before']:.4f} -> "
        f"{summary['mean_reliability_after']:.4f}",
    )
    print(
        "post actions:",
        summary["post_intervention_action_rates"],
    )
    print("saved:", out)


if __name__ == "__main__":
    freeze_support()
    main()
