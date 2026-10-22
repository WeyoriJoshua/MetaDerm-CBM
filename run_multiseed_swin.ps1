param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int[]]$Seeds = @(2026,2027,2028,2029,2030)
)

$ErrorActionPreference = "Stop"

Write-Host "=== MetaDerm-CBM Swin-Tiny frozen sensitivity study ==="

if (-not (Test-Path "configs/swin_tiny_final.yaml")) {
  python scripts/18_prepare_swin_sensitivity.py
  if ($LASTEXITCODE -ne 0) {
    throw "Swin configuration preparation failed."
  }
}

python scripts/00_audit_data.py `
  --config configs/base.yaml `
  --override configs/swin_tiny_final.yaml `
  --data-root $DataRoot `
  --seed 2026

foreach ($Seed in $Seeds) {
  .\run_pipeline_swin.ps1 `
    -DataRoot $DataRoot `
    -Seed $Seed
}

python scripts/15_aggregate_final_seeds.py `
  --results-root "results/swin_tiny_final"

python scripts/16_cross_seed_selective_analysis.py `
  --results-root "results/swin_tiny_final" `
  --seeds "2026,2027,2028,2029,2030"

python scripts/17_publication_tables_figures.py `
  --results-root "results/swin_tiny_final" `
  --n-boot 10000 `
  --seed 2026

Write-Host "=== Swin-Tiny study complete ==="
