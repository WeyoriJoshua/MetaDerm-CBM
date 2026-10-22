param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int]$Seed = 2026
)
$ErrorActionPreference = "Stop"
python scripts/00_audit_data.py --config configs/base.yaml --data-root $DataRoot --seed $Seed
python scripts/01_train_concepts.py --config configs/base.yaml --data-root $DataRoot --seed $Seed
python scripts/02_train_blackbox.py --config configs/base.yaml --data-root $DataRoot --seed $Seed
python scripts/03_fit_cbm_baselines.py --config configs/base.yaml --data-root $DataRoot --seed $Seed --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${Seed}_best.pt"
python scripts/04_calibrate.py --config configs/base.yaml --data-root $DataRoot --seed $Seed --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${Seed}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${Seed}_best.pt"
python scripts/05_evaluate_metaderm.py --config configs/base.yaml --data-root $DataRoot --seed $Seed --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${Seed}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${Seed}_best.pt" --calibration "results/efficientnet_b3/calibration/calibration_seed${Seed}.json"
python scripts/06_run_intervention.py --config configs/base.yaml --data-root $DataRoot --seed $Seed --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${Seed}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${Seed}_best.pt"
python scripts/07_run_ablation.py --config configs/base.yaml --data-root $DataRoot --seed $Seed --ablation all --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${Seed}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${Seed}_best.pt" --calibration "results/efficientnet_b3/calibration/calibration_seed${Seed}.json"
python scripts/08_bootstrap_statistics.py --predictions results/efficientnet_b3/final/seed${Seed}/test_predictions.csv --seed $Seed
python scripts/09_make_publication_tables.py --results-root results/efficientnet_b3
