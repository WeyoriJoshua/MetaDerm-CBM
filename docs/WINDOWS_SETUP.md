# Windows setup and first run

From PowerShell, inside the repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the appropriate PyTorch build for your NVIDIA/CUDA environment if using a GPU, then:

```powershell
pip install -r requirements.txt
pip install -e .
```

Audit the dataset before training:

```powershell
python scripts/00_audit_data.py --config configs/base.yaml --data-root "D:\PATH\TO\release_v0"
```

If the audit reports zero split overlap, zero missing/corrupt images, and no unknown concept states, run a complete single-seed pipeline:

```powershell
.\run_pipeline.ps1 -DataRoot "D:\PATH\TO\release_v0" -Seed 2026
```

For the manuscript, repeat model training for seeds 2027–2030 rather than treating a single seed as the final result.
