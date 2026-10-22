from multiprocessing import freeze_support
from _common import parser, setup
from pathlib import Path
import json
from metaderm.data.audit import audit_dataset
from metaderm.reproducibility import environment_manifest, write_manifest
from metaderm.utils.io import ensure_dir, save_json


def main():
    p=parser("Audit Derm7pt before any training"); p.add_argument("--skip-image-check",action="store_true"); args=p.parse_args(); cfg,seed,device=setup(args)
    out=ensure_dir(Path(cfg["project"]["output_root"])/"audit")
    a=audit_dataset(cfg,check_images=not args.skip_image_check)
    a["split_summary"].to_csv(out/"split_summary.csv",index=False); a["concept_summary"].to_csv(out/"concept_summary.csv",index=False); a["profiles"].to_csv(out/"training_profile_reliability.csv",index=False); a["image_errors"].to_csv(out/"image_errors.csv",index=False)
    save_json({"n_total":a["n_total"],"overlaps":a["overlaps"],"fingerprint":a["fingerprint"]},out/"audit_summary.json"); write_manifest(environment_manifest(),out/"environment.json")
    print(a["split_summary"].to_string(index=False)); print("training profiles:",len(a["profiles"]),"inconsistent:",int(a["profiles"]["inconsistent"].sum())); print("image errors:",len(a["image_errors"])); print("saved:",out)


if __name__ == "__main__":
    freeze_support()
    main()
