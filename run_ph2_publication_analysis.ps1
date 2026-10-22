param(
  [int]$BootstrapIterations = 10000
)

$ErrorActionPreference = "Stop"

python scripts/31_ph2_publication_analysis.py `
  --external-root "results/ph2_external/efficientnet_b3" `
  --internal-root "results/efficientnet_b3_final" `
  --output-root "results/ph2_external/efficientnet_b3/publication" `
  --bootstrap-iterations $BootstrapIterations `
  --seed 2026

if ($LASTEXITCODE -ne 0) {
  throw "PH2 publication analysis failed."
}

Write-Host "=== PH2 publication analysis finished ==="
