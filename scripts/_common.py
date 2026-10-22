from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from metaderm.config import load_config
from metaderm.reproducibility import seed_everything


def parser(description):
    p=argparse.ArgumentParser(description=description); p.add_argument("--config",default="configs/base.yaml"); p.add_argument("--override",action="append",default=[],help="Additional YAML override; may be repeated"); p.add_argument("--data-root",default=None); p.add_argument("--seed",type=int,default=None); return p

def setup(args):
    cfg=load_config(args.config, *(args.override or []))
    if args.data_root: cfg["data"]["root"]=args.data_root
    seed=int(args.seed if args.seed is not None else cfg["project"]["seed"]); cfg["project"]["seed"]=seed; seed_everything(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:",device); return cfg,seed,device
