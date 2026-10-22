from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import pandas as pd
from metaderm.data.derm7pt import make_loader
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.training.setup import training_pos_weight
from metaderm.training.engine import train_blackbox
from metaderm.reproducibility import environment_manifest, write_manifest
from metaderm.utils.io import ensure_dir


def main():
    args=parser("Train independent black-box diagnostic model").parse_args(); cfg,seed,device=setup(args); model=BlackBoxDiagnosisModel(cfg).to(device); pw=training_pos_weight(cfg); tr=make_loader(cfg,"train"); va=make_loader(cfg,"valid",shuffle=False); hist,ckpt=train_blackbox(model,tr,va,cfg,pw,device,seed)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"training"/f"blackbox_seed{seed}"); pd.DataFrame(hist).to_csv(out/"history.csv",index=False); write_manifest(environment_manifest(),out/"environment.json"); print("best checkpoint:",ckpt)


if __name__ == "__main__":
    freeze_support()
    main()
