
param(
  [Parameter(Mandatory=$true)][string]$Derm7ptRoot
)

$ErrorActionPreference = "Stop"

python scripts/23_run_ph2_external_validation.py `
  --ph2-metadata "data_external/ph2/ph2_canonical.csv" `
  --derm7pt-root $Derm7ptRoot `
  --config "configs/base.yaml" `
  --overlay "configs/final_design.yaml" `
  --checkpoint-root "checkpoints/efficientnet_b3_final" `
  --internal-results-root "results/efficientnet_b3_final" `
  --output-root "results/ph2_external/efficientnet_b3" `
  --seeds "2026,2027,2028,2029,2030" `
  --batch-size 16

if ($LASTEXITCODE -ne 0) {
  throw "PH2 external validation failed."
}

python scripts/24_compare_internal_external.py `
  --internal-root "results/efficientnet_b3_final" `
  --external-root "results/ph2_external/efficientnet_b3"

Write-Host "=== PH2 external validation and comparison complete ==="
