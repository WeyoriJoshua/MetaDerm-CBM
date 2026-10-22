param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [int[]]$Seeds = @(2026,2027,2028,2029,2030)
)
$ErrorActionPreference = "Stop"
foreach ($Seed in $Seeds) {
  Write-Host "=== MetaDerm-CBM seed $Seed ==="
  .\run_pipeline.ps1 -DataRoot $DataRoot -Seed $Seed
}
