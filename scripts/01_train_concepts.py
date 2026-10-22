from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import pandas as pd
from metaderm.data.derm7pt import make_loader
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.training.setup import training_class_weights
from metaderm.training.engine import train_concepts
from metaderm.reproducibility import environment_manifest, write_manifest
from metaderm.utils.io import ensure_dir


def main():
    args=parser("Train seven-head concept perception model").parse_args(); cfg,seed,device=setup(args); model=ConceptPerceptionModel(cfg).to(device); w=training_class_weights(cfg); tr=make_loader(cfg,"train"); va=make_loader(cfg,"valid",shuffle=False); hist,ckpt=train_concepts(model,tr,va,cfg,w,device,seed)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"training"/f"concepts_seed{seed}"); pd.DataFrame(hist).to_csv(out/"history.csv",index=False); write_manifest(environment_manifest(),out/"environment.json"); print("best checkpoint:",ckpt)


if __name__ == "__main__":
    freeze_support()
    main()
