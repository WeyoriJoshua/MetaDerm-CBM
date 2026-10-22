param(
  [Parameter(Mandatory=$true)][string]$Derm7ptRoot
)

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Representation visualization ==="
python scripts/32_representation_visualization.py `
  --derm7pt-root $Derm7ptRoot `
  --ph2-metadata "data_external/ph2/ph2_canonical.csv" `
  --config "configs/base.yaml" `
  --overlay "configs/final_design.yaml" `
  --checkpoint "checkpoints/efficientnet_b3_final/blackbox_seed2026_best.pt" `
  --output-root "results/interpretability/representation_seed2026" `
  --seed 2026

if ($LASTEXITCODE -ne 0) {
  throw "Representation visualization failed."
}

Write-Host "=== 2. PH2 Grad-CAM ==="
python scripts/33_gradcam_ph2.py `
  --ph2-metadata "data_external/ph2/ph2_canonical.csv" `
  --predictions "results/ph2_external/efficientnet_b3/seed2026/ph2_predictions.csv" `
  --config "configs/base.yaml" `
  --overlay "configs/final_design.yaml" `
  --checkpoint "checkpoints/efficientnet_b3_final/blackbox_seed2026_best.pt" `
  --output-root "results/interpretability/gradcam_ph2_seed2026"

if ($LASTEXITCODE -ne 0) {
  throw "Grad-CAM analysis failed."
}

Write-Host "=== Representation and Grad-CAM analyses complete ==="
