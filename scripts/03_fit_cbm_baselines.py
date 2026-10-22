from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader, load_metadata, load_split_indices, concept_mappings
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.training.collect import collect_concept_outputs
from metaderm.evaluation.metrics import binary_metrics
from metaderm.utils.io import load_checkpoint, ensure_dir, save_json


def main():
    p=parser("Fit and evaluate learned CBM baselines")
    p.add_argument("--concept-checkpoint",required=True)
    args=p.parse_args(); cfg,seed,device=setup(args)
    model=ConceptPerceptionModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,model,device)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"cbm_baselines"/f"seed{seed}")

    features={}; labels={}; indices={}
    for split in ["train","valid","test"]:
        c=collect_concept_outputs(model,make_loader(cfg,split,shuffle=False),device)
        probs=[torch.softmax(c["logits"][name],1).numpy() for name in CONCEPT_COLUMNS]
        features[split]=np.concatenate(probs,axis=1)
        labels[split]=c["labels"].numpy().astype(int)
        indices[split]=c["source_indices"].numpy().astype(int)

    pred_clf=LogisticRegression(max_iter=5000,class_weight="balanced",solver="lbfgs",random_state=seed).fit(features["train"],labels["train"])
    joblib.dump(pred_clf,out/"predicted_concept_logreg.joblib")

    meta=load_metadata(cfg); maps=concept_mappings(cfg); widths=[len(cfg["data"]["concepts"][c]) for c in CONCEPT_COLUMNS]
    def oracle(split):
        part=meta.iloc[load_split_indices(cfg,split)]
        arr=[]
        for _,row in part.iterrows():
            v=[]
            for c,w in zip(CONCEPT_COLUMNS,widths):
                one=np.zeros(w); one[maps[c][str(row[c])]]=1; v.extend(one)
            arr.append(v)
        return np.asarray(arr),part["binary_label"].to_numpy().astype(int)
    Xtr,ytr=oracle("train")
    oracle_clf=LogisticRegression(max_iter=5000,class_weight="balanced",solver="lbfgs",random_state=seed).fit(Xtr,ytr)
    joblib.dump(oracle_clf,out/"oracle_concept_logreg.joblib")

    metrics={}; rows=[]
    for split in ["valid","test"]:
        p_pred=pred_clf.predict_proba(features[split])[:,1]
        Xo,yo=oracle(split); p_oracle=oracle_clf.predict_proba(Xo)[:,1]
        metrics[split]={"predicted_concept_cbm":binary_metrics(labels[split],p_pred),"oracle_concept_cbm":binary_metrics(yo,p_oracle)}
        rows.append(pd.DataFrame({"source_index":indices[split],"label":labels[split],"predicted_concept_cbm_prob":p_pred,"oracle_concept_cbm_prob":p_oracle,"split":split}))
    pd.concat(rows,ignore_index=True).to_csv(out/"cbm_predictions.csv",index=False)
    save_json(metrics,out/"cbm_metrics.json")
    save_json({"feature_dim":int(features['train'].shape[1]),"concept_checkpoint":args.concept_checkpoint},out/"manifest.json")
    print(metrics["test"]); print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
