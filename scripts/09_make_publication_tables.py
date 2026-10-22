from multiprocessing import freeze_support
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results-root",default="results"); args=p.parse_args()
    root=Path(args.results_root); out=root/"publication"/"tables"; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed_dir in sorted((root/"final").glob("seed*")):
        metrics_path=seed_dir/"metrics.json"
        if not metrics_path.exists(): continue
        seed=int(re.sub(r"\D","",seed_dir.name) or -1); m=json.loads(metrics_path.read_text())
        for model in ["blackbox","soft_reasoner"]:
            if model in m:
                rows.append({"seed":seed,"model":model,**m[model],"predict_coverage":m.get("coverage_predict"),"query_rate":m.get("query_rate"),"abstain_rate":m.get("abstain_rate")})
    if rows:
        per_seed=pd.DataFrame(rows); per_seed.to_csv(out/"diagnostic_results_per_seed.csv",index=False)
        numeric=[c for c in per_seed.columns if c not in {"seed","model"}]
        summary=per_seed.groupby("model")[numeric].agg(["mean","std"])
        summary.columns=[f"{a}_{b}" for a,b in summary.columns]; summary=summary.reset_index(); summary.to_csv(out/"diagnostic_results_mean_sd.csv",index=False)
        (out/"diagnostic_results_mean_sd.tex").write_text(summary.to_latex(index=False,float_format=lambda x:f"{x:.4f}"),encoding="utf-8")

    concept_rows=[]
    for seed_dir in sorted((root/"final").glob("seed*")):
        f=seed_dir/"concept_metrics.csv"
        if f.exists():
            seed=int(re.sub(r"\D","",seed_dir.name) or -1); d=pd.read_csv(f); d.insert(0,"seed",seed); concept_rows.append(d)
    if concept_rows:
        cdf=pd.concat(concept_rows,ignore_index=True); cdf.to_csv(out/"concept_results_per_seed.csv",index=False)
        cols=[c for c in cdf.columns if c not in {"seed","concept"}]
        csum=cdf.groupby("concept")[cols].agg(["mean","std"]); csum.columns=[f"{a}_{b}" for a,b in csum.columns]; csum=csum.reset_index(); csum.to_csv(out/"concept_results_mean_sd.csv",index=False); (out/"concept_results_mean_sd.tex").write_text(csum.to_latex(index=False,float_format=lambda x:f"{x:.4f}"),encoding="utf-8")

    ablation=[]
    for f in sorted((root/"ablation").glob("seed*/ablation_summary.csv")):
        seed=int(re.sub(r"\D","",f.parent.name) or -1); d=pd.read_csv(f); d.insert(0,"seed",seed); ablation.append(d)
    if ablation:
        adf=pd.concat(ablation,ignore_index=True); adf.to_csv(out/"ablation_per_seed.csv",index=False)
        cols=[c for c in adf.columns if c not in {"seed","ablation"}]
        asum=adf.groupby("ablation")[cols].agg(["mean","std"]); asum.columns=[f"{a}_{b}" for a,b in asum.columns]; asum=asum.reset_index(); asum.to_csv(out/"ablation_mean_sd.csv",index=False)
    print("publication tables:",out)


if __name__ == "__main__":
    freeze_support()
    main()
