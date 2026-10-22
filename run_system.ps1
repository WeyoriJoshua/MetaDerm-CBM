param(
  [Parameter(Mandatory=$true)][string]$Derm7ptRoot,
  [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$env:METADERM_DERM7PT_ROOT = $Derm7ptRoot

python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed in the active environment."
    Write-Host "Install it with: python -m pip install streamlit"
    exit 1
}

Write-Host "Starting MetaDerm-CBM decision-support prototype..."
Write-Host "Derm7pt root: $Derm7ptRoot"
Write-Host "Port: $Port"

python -m streamlit run system\app.py `
  --server.port $Port `
  --server.address "localhost"
