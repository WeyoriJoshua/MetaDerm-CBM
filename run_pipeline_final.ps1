param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"

$Config = "configs/base.yaml"
$FinalOverlay = "configs/final_design.yaml"
$CheckpointRoot = "checkpoints/efficientnet_b3_final"
$ResultsRoot = "results/efficientnet_b3_final"

Write-Host "=== MetaDerm-CBM FINAL seed $Seed ==="

python scripts/01_train_concepts.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed

python scripts/02_train_blackbox.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed

$ConceptCkpt = "$CheckpointRoot/concepts_seed${Seed}_best.pt"
$BlackboxCkpt = "$CheckpointRoot/blackbox_seed${Seed}_best.pt"

python scripts/04_calibrate.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt

$Calibration = "$ResultsRoot/calibration/calibration_seed${Seed}.json"

python scripts/05_evaluate_metaderm.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt `
  --calibration $Calibration

python scripts/03_fit_cbm_baselines.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt

python scripts/06_run_intervention_final.py `
  --config $Config `
  --override $FinalOverlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt `
  --calibration $Calibration

python scripts/08_bootstrap_statistics.py `
  --predictions "$ResultsRoot/final/seed${Seed}/test_predictions.csv" `
  --n-boot 10000 `
  --seed $Seed

Write-Host "=== Completed FINAL seed $Seed ==="
