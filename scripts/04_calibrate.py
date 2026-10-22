from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
from metaderm.constants import CONCEPT_COLUMNS
from metaderm.data.derm7pt import make_loader
from metaderm.models.concept_model import ConceptPerceptionModel
from metaderm.models.diagnosis_model import BlackBoxDiagnosisModel
from metaderm.training.collect import collect_concept_outputs, collect_blackbox_outputs
from metaderm.uncertainty.calibration import TemperatureScaler
from metaderm.utils.io import load_checkpoint, ensure_dir, save_json


def main():
    p=parser("Fit validation-only temperature scaling"); p.add_argument("--concept-checkpoint",required=True); p.add_argument("--blackbox-checkpoint",required=True); args=p.parse_args(); cfg,seed,device=setup(args)
    cm=ConceptPerceptionModel(cfg).to(device); bm=BlackBoxDiagnosisModel(cfg).to(device); load_checkpoint(args.concept_checkpoint,cm,device); load_checkpoint(args.blackbox_checkpoint,bm,device); va=make_loader(cfg,"valid",shuffle=False); c=collect_concept_outputs(cm,va,device); b=collect_blackbox_outputs(bm,va,device)
    temps={}
    for name in CONCEPT_COLUMNS:
        sc=TemperatureScaler(); temps[name]=sc.fit_multiclass(c["logits"][name],c["targets"][name],cfg["calibration"]["max_iter"])
    sc=TemperatureScaler(); bb=sc.fit_binary(b["logits"],b["labels"],cfg["calibration"]["max_iter"])
    out=ensure_dir(Path(cfg["project"]["output_root"])/"calibration"); path=out/f"calibration_seed{seed}.json"; save_json({"concept_temperatures":temps,"blackbox_temperature":bb},path); print(path)


if __name__ == "__main__":
    freeze_support()
    main()
