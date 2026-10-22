from multiprocessing import freeze_support
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from metaderm.evaluation.stats import bootstrap_metric, paired_bootstrap_difference
from metaderm.utils.io import save_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--n-boot",type=int,default=10000); p.add_argument("--seed",type=int,default=2026); args=p.parse_args(); df=pd.read_csv(args.predictions); y=df.label.to_numpy(); out={"blackbox_auroc":bootstrap_metric(y,df.blackbox_prob.to_numpy(),roc_auc_score,args.n_boot,args.seed),"reason_auroc":bootstrap_metric(y,df.reason_prob.to_numpy(),roc_auc_score,args.n_boot,args.seed),"auroc_difference_bb_minus_reason":paired_bootstrap_difference(y,df.blackbox_prob.to_numpy(),df.reason_prob.to_numpy(),roc_auc_score,args.n_boot,args.seed)}; path=Path(args.predictions).parent/"bootstrap_statistics.json"; save_json(out,path); print(out)


if __name__ == "__main__":
    freeze_support()
    main()
