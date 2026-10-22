from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader, load_metadata, load_split_indices
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.evaluation.pipeline import evaluate_components
from metaderm.evaluation.metrics import binary_metrics
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.reasoning.consistency import hard_profile_from_probs, build_profile_map, lookup_profile_reliability
from metaderm.reasoning.metacognition import metacognitive_features, reliability_score
from metaderm.utils.io import load_checkpoint, load_json, ensure_dir


def main():
    p=parser("Concept-probability corruption robustness experiment")
    p.add_argument("--concept-checkpoint",required=True); p.add_argument("--blackbox-checkpoint",required=True); p.add_argument("--calibration",default=None); p.add_argument("--rates",default="0.05,0.10,0.20,0.30"); p.add_argument("--strength",type=float,default=0.75)
    args=p.parse_args(); cfg,seed,device=setup(args); cal=load_json(args.calibration) if args.calibration else None; cm=ConceptPerceptionModel(cfg).to(device); bm=BlackBoxDiagnosisModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,cm,device); load_checkpoint(args.blackbox_checkpoint,bm,device); base=evaluate_components(cfg,cm,bm,make_loader(cfg,"test",shuffle=False),device,cal); reasoner=SoftSevenPointReasoner(cfg); rng=np.random.default_rng(seed); rows=[]; n=len(base["labels"])
    meta=load_metadata(cfg); pmap=build_profile_map(meta.iloc[load_split_indices(cfg,"train")],cfg)
    for rate in [float(x) for x in args.rates.split(',')]:
        cp={c:p.copy() for c,p in base["concept_probs"].items()}; corrupted=np.zeros(n,dtype=int); m=max(1,int(round(rate*n))); cases=rng.choice(n,size=m,replace=False); corrupted[cases]=1
        for i in cases:
            c=CONCEPT_COLUMNS[int(rng.integers(0,len(CONCEPT_COLUMNS)))]; k=cp[c].shape[1]; uniform=np.full(k,1.0/k); cp[c][i]=(1-args.strength)*cp[c][i]+args.strength*uniform
        reason=reasoner.predict(cp); profiles=hard_profile_from_probs(cp); pr=lookup_profile_reliability(profiles,pmap,cfg["metacognition"].get("unseen_profile_reliability",0.0)); feat=metacognitive_features(cp,base["blackbox_prob"],reason["probability"],pr); rel=reliability_score(feat,cfg); detect_auc=roc_auc_score(corrupted,1-rel) if len(np.unique(corrupted))==2 else float('nan')
        met=binary_metrics(base["labels"],reason["probability"]); rows.append({"corruption_rate":rate,"strength":args.strength,"reason_auroc":met["auroc"],"reason_f1":met["f1"],"mean_reliability":float(rel.mean()),"corruption_detection_auroc":float(detect_auc)})
    out=ensure_dir(Path(cfg["project"]["output_root"])/"robustness"/f"seed{seed}"); pd.DataFrame(rows).to_csv(out/"concept_noise_robustness.csv",index=False); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
