from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd

from _common import parser, setup
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.intervention import (
    apply_oracle_correction,
    reasoning_impact,
    select_query_concept,
)
from metaderm.reasoning.metacognition import (
    assign_actions,
    select_three_way_thresholds,
)
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.uncertainty.entropy import normalized_categorical_entropy
from metaderm.utils.io import ensure_dir, load_checkpoint, load_json


def _select_concept(policy, reasoner, concept_probs, i, rng):
    if policy == "proposed":
        best, _ = select_query_concept(reasoner, concept_probs, i)
        return best[1]

    if policy == "uncertainty_only":
        scored = []
        for c in CONCEPT_COLUMNS:
            u = float(
                normalized_categorical_entropy(
                    concept_probs[c][i:i+1]
                )[0]
            )
            scored.append((u, c))
        return max(scored)[1]

    if policy == "impact_only":
        scored = [
            (reasoning_impact(reasoner, concept_probs, i, c), c)
            for c in CONCEPT_COLUMNS
        ]
        return max(scored)[1]

    if policy == "random":
        return str(rng.choice(CONCEPT_COLUMNS))

    raise ValueError(policy)


def _run_policy(
    policy,
    reasoner,
    test,
    query_indices,
    rng,
):
    before_correct = []
    after_correct = []
    recovered = 0
    introduced = 0
    selected_wrong = 0
    selected = []

    for i in query_indices:
        concept = _select_concept(
            policy, reasoner, test["concept_probs"], int(i), rng
        )
        selected.append(concept)

        pred_state = int(test["concept_probs"][concept][i].argmax())
        true_state = int(test["concept_raw"]["targets"][concept][i])
        selected_wrong += int(pred_state != true_state)

        corrected = apply_oracle_correction(
            test["concept_probs"],
            int(i),
            concept,
            true_state,
        )

        label = int(test["labels"][i])
        before_prob = float(test["reason"]["probability"][i])
        after_prob = float(reasoner.predict(corrected)["probability"][0])

        b = int((before_prob >= 0.5) == label)
        a = int((after_prob >= 0.5) == label)
        before_correct.append(b)
        after_correct.append(a)
        recovered += int(b == 0 and a == 1)
        introduced += int(b == 1 and a == 0)

    n = len(query_indices)
    return {
        "policy": policy,
        "n_query": int(n),
        "selected_concept_wrong_rate": float(selected_wrong / n),
        "reason_accuracy_before": float(np.mean(before_correct)),
        "reason_accuracy_after": float(np.mean(after_correct)),
        "absolute_accuracy_gain": float(
            np.mean(after_correct) - np.mean(before_correct)
        ),
        "errors_recovered": int(recovered),
        "errors_introduced": int(introduced),
        "net_error_recovery": int(recovered - introduced),
        "selected_concepts": selected,
    }


def main():
    p = parser("Compare concept-query selection policies on QUERY cases")
    p.add_argument("--concept-checkpoint", required=True)
    p.add_argument("--blackbox-checkpoint", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--random-repeats", type=int, default=100)
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
    actions = assign_actions(test["reliability"], tau_low, tau_high)
    query_indices = np.flatnonzero(actions == "query")
    reasoner = SoftSevenPointReasoner(cfg)

    deterministic = []
    for policy in ["proposed", "uncertainty_only", "impact_only"]:
        deterministic.append(
            _run_policy(
                policy,
                reasoner,
                test,
                query_indices,
                np.random.default_rng(seed),
            )
        )

    random_rows = []
    for r in range(args.random_repeats):
        row = _run_policy(
            "random",
            reasoner,
            test,
            query_indices,
            np.random.default_rng(seed + r),
        )
        row["repeat"] = r
        random_rows.append(row)

    out = ensure_dir(
        Path(cfg["project"]["output_root"])
        / "query_policy_baselines"
        / f"seed{seed}"
    )

    # Save deterministic policies without the per-case selection list.
    det_df = pd.DataFrame([
        {k: v for k, v in x.items() if k != "selected_concepts"}
        for x in deterministic
    ])
    det_df.to_csv(out / "deterministic_policies.csv", index=False)

    rand_df = pd.DataFrame([
        {k: v for k, v in x.items() if k != "selected_concepts"}
        for x in random_rows
    ])
    rand_df.to_csv(out / "random_policy_repeats.csv", index=False)

    random_summary = {
        "policy": "random",
        "n_query": int(len(query_indices)),
        "selected_concept_wrong_rate_mean": float(
            rand_df["selected_concept_wrong_rate"].mean()
        ),
        "reason_accuracy_after_mean": float(
            rand_df["reason_accuracy_after"].mean()
        ),
        "reason_accuracy_after_ci_low": float(
            rand_df["reason_accuracy_after"].quantile(0.025)
        ),
        "reason_accuracy_after_ci_high": float(
            rand_df["reason_accuracy_after"].quantile(0.975)
        ),
        "absolute_accuracy_gain_mean": float(
            rand_df["absolute_accuracy_gain"].mean()
        ),
        "net_error_recovery_mean": float(
            rand_df["net_error_recovery"].mean()
        ),
    }

    print("Deterministic query policies")
    print(det_df.to_string(index=False))
    print("\nRandom baseline over", args.random_repeats, "repeats")
    for k, v in random_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    freeze_support()
    main()
