from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.external_manifest import ExternalManifestDataset
from metaderm.data.derm7pt import load_metadata, load_split_indices
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.reasoning.consistency import build_profile_map, hard_profile_from_probs, lookup_profile_reliability
from metaderm.reasoning.metacognition import metacognitive_features, reliability_score, risk_coverage_curve
from metaderm.evaluation.metrics import binary_metrics
from metaderm.evaluation.plots import plot_risk_coverage
from metaderm.utils.io import load_checkpoint, load_json, ensure_dir, save_json


def main():
    p=parser("Frozen external diagnostic validation")
    p.add_argument("--manifest",required=True); p.add_argument("--image-root",required=True); p.add_argument("--concept-checkpoint",required=True); p.add_argument("--blackbox-checkpoint",required=True); p.add_argument("--calibration",default=None)
    args=p.parse_args(); cfg,seed,device=setup(args); cal=load_json(args.calibration) if args.calibration else {}; ct=cal.get("concept_temperatures",{}); bt=float(cal.get("blackbox_temperature",1.0))
    ds=ExternalManifestDataset(args.manifest,args.image_root,cfg["data"]["image_size"]); loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=False,num_workers=cfg["data"].get("num_workers",0))
    cm=ConceptPerceptionModel(cfg).to(device); bm=BlackBoxDiagnosisModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,cm,device); load_checkpoint(args.blackbox_checkpoint,bm,device); cm.eval(); bm.eval(); cp={c:[] for c in CONCEPT_COLUMNS}; bb=[]; ys=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["image"].to(device); co=cm(x)["logits"]; bo=bm(x)["logit"]
            for c in CONCEPT_COLUMNS: cp[c].append(torch.softmax(co[c]/float(ct.get(c,1.0)),1).cpu().numpy())
            bb.append(torch.sigmoid(bo/bt).cpu().numpy()); ys.append(batch["label"].numpy())
    cp={c:np.concatenate(v) for c,v in cp.items()}; bb=np.concatenate(bb); y=np.concatenate(ys).astype(int); reasoner=SoftSevenPointReasoner(cfg); reason=reasoner.predict(cp)
    meta=load_metadata(cfg); tr=meta.iloc[load_split_indices(cfg,"train")]; pmap=build_profile_map(tr,cfg); profiles=hard_profile_from_probs(cp); pr=lookup_profile_reliability(profiles,pmap,cfg["metacognition"].get("unseen_profile_reliability",0.0)); feat=metacognitive_features(cp,bb,reason["probability"],pr); rel=reliability_score(feat,cfg); pred=(bb>=.5).astype(int); curve=risk_coverage_curve(y,pred,rel)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"external"/f"seed{seed}"); save_json({"blackbox":binary_metrics(y,bb),"soft_reasoner":binary_metrics(y,reason["probability"]),"n":len(y)},out/"metrics.json"); pd.DataFrame({"label":y,"blackbox_prob":bb,"reason_prob":reason["probability"],"reliability":rel,"profile_reliability":pr,"concept_uncertainty":feat["concept_uncertainty"],"disagreement":feat["neural_symbolic_disagreement"]}).to_csv(out/"predictions.csv",index=False); pd.DataFrame(curve,columns=["coverage","risk","threshold"]).to_csv(out/"risk_coverage.csv",index=False); plot_risk_coverage(curve,out/"risk_coverage.png"); print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
