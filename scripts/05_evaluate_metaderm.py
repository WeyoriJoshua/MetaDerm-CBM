from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import numpy as np, pandas as pd
from metaderm.data.derm7pt import make_loader
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.evaluation.metrics import binary_metrics
from metaderm.evaluation.concepts import concept_metrics
from metaderm.evaluation.plots import plot_risk_coverage, plot_reliability_hist
from metaderm.reasoning.metacognition import select_three_way_thresholds, assign_actions, risk_coverage_curve
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.utils.io import load_checkpoint, load_json, ensure_dir, save_json


def main():
    p=parser("Final MetaDerm-CBM internal evaluation"); p.add_argument("--concept-checkpoint",required=True); p.add_argument("--blackbox-checkpoint",required=True); p.add_argument("--calibration",default=None); args=p.parse_args(); cfg,seed,device=setup(args); calibration=load_json(args.calibration) if args.calibration else None
    cm=ConceptPerceptionModel(cfg).to(device); bm=BlackBoxDiagnosisModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,cm,device); load_checkpoint(args.blackbox_checkpoint,bm,device)
    # validation selects action thresholds only
    val=evaluate_components(cfg,cm,bm,make_loader(cfg,"valid",shuffle=False),device,calibration); val_pred=(val["blackbox_prob"]>=.5).astype(int); tau_low,tau_high=select_three_way_thresholds(val["labels"],val_pred,val["reliability"],cfg["metacognition"]["target_accepted_risk"],cfg["metacognition"]["minimum_coverage"],cfg["metacognition"]["abstain_fraction_of_deferred"])
    # test is untouched until thresholds fixed
    test=evaluate_components(cfg,cm,bm,make_loader(cfg,"test",shuffle=False),device,calibration); y=test["labels"]; bb=test["blackbox_prob"]; rp=test["reason"]["probability"]; r=test["reliability"]; pred=(bb>=.5).astype(int); actions=assign_actions(r,tau_low,tau_high); curve=risk_coverage_curve(y,pred,r)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"final"/f"seed{seed}"); figs=ensure_dir(out/"figures")
    metrics={"blackbox":binary_metrics(y,bb),"soft_reasoner":binary_metrics(y,rp),"tau_low":tau_low,"tau_high":tau_high,"coverage_predict":float(np.mean(actions=="predict")),"query_rate":float(np.mean(actions=="query")),"abstain_rate":float(np.mean(actions=="abstain"))}; save_json(metrics,out/"metrics.json")
    cm_df,cm_summary=concept_metrics({c:test["concept_raw"]["targets"][c].numpy() for c in CONCEPT_COLUMNS},test["concept_probs"],cfg); cm_df.to_csv(out/"concept_metrics.csv",index=False); save_json(cm_summary,out/"concept_summary.json")
    pd.DataFrame(curve,columns=["coverage","risk","threshold"]).to_csv(out/"risk_coverage.csv",index=False)
    rows={"source_index":test["source_indices"],"label":y,"blackbox_prob":bb,"reason_prob":rp,"reliability":r,"profile_reliability":test["profile_reliability"],"concept_uncertainty":test["features"]["concept_uncertainty"],"diagnosis_uncertainty":test["features"]["diagnosis_uncertainty"],"neural_symbolic_disagreement":test["features"]["neural_symbolic_disagreement"],"action":actions}
    for c in CONCEPT_COLUMNS:
        rows[f"{c}_true"]=test["concept_raw"]["targets"][c].numpy(); rows[f"{c}_pred"]=test["concept_probs"][c].argmax(1); rows[f"{c}_entropy"]=test["features"]["per_concept_entropy"][:,CONCEPT_COLUMNS.index(c)]
    pd.DataFrame(rows).to_csv(out/"test_predictions.csv",index=False); pd.Series(actions).value_counts().rename_axis("action").reset_index(name="n").to_csv(out/"action_summary.csv",index=False)
    plot_risk_coverage(curve,figs/"risk_coverage.png"); plot_reliability_hist(r,y,pred,figs/"reliability_correctness.png"); print(metrics); print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
