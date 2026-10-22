param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"

$Config = "configs/base.yaml"
$Overlay = "configs/swin_tiny_final.yaml"
$CheckpointRoot = "checkpoints/swin_tiny_final"
$ResultsRoot = "results/swin_tiny_final"

Write-Host "=== MetaDerm-CBM SWIN-TINY sensitivity seed $Seed ==="

python scripts/01_train_concepts.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed

python scripts/02_train_blackbox.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed

$ConceptCkpt = "$CheckpointRoot/concepts_seed${Seed}_best.pt"
$BlackboxCkpt = "$CheckpointRoot/blackbox_seed${Seed}_best.pt"

python scripts/04_calibrate.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt

$Calibration = "$ResultsRoot/calibration/calibration_seed${Seed}.json"

python scripts/05_evaluate_metaderm.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt `
  --calibration $Calibration

python scripts/03_fit_cbm_baselines.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt

python scripts/06_run_intervention_final.py `
  --config $Config `
  --override $Overlay `
  --data-root $DataRoot `
  --seed $Seed `
  --concept-checkpoint $ConceptCkpt `
  --blackbox-checkpoint $BlackboxCkpt `
  --calibration $Calibration

python scripts/08_bootstrap_statistics.py `
  --predictions "$ResultsRoot/final/seed${Seed}/test_predictions.csv" `
  --n-boot 10000 `
  --seed $Seed

Write-Host "=== Completed SWIN-TINY seed $Seed ==="
