param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int[]]$Seeds = @(2026,2027,2028,2029,2030)
)

$ErrorActionPreference = "Stop"

Write-Host "=== MetaDerm-CBM frozen-design five-seed study ==="
Write-Host "DataRoot: $DataRoot"

# One audit is sufficient because dataset integrity is seed-independent.
python scripts/00_audit_data.py `
  --config configs/base.yaml `
  --override configs/final_design.yaml `
  --data-root $DataRoot `
  --seed 2026

foreach ($Seed in $Seeds) {
  .\run_pipeline_final.ps1 `
    -DataRoot $DataRoot `
    -Seed $Seed
}

python scripts/15_aggregate_final_seeds.py `
  --results-root "results/efficientnet_b3_final"

Write-Host "=== All final seeds completed and aggregated ==="
