# START HERE — MetaDerm-CBM

## First real-data run

1. Extract the repository.
2. Create/activate a Python 3.11 or 3.12 virtual environment.
3. Install PyTorch for your GPU/CUDA environment, then run:

```powershell
pip install -r requirements.txt
pip install -e .
```

4. Do **not** move the Derm7pt images into this repository. Keep the authorized dataset in its own folder.
5. Confirm that your extracted Derm7pt root contains:

```text
images/
meta/meta.csv
meta/train_indexes.csv
meta/valid_indexes.csv
meta/test_indexes.csv
```

6. Run the mandatory audit first:

```powershell
python scripts/00_audit_data.py --config configs/base.yaml --data-root "D:\YOUR\PATH\release_v0"
```

7. Only if the audit reports zero split overlap, zero missing/corrupt images, and no unknown concept states, begin training:

```powershell
.\run_pipeline.ps1 -DataRoot "D:\YOUR\PATH\release_v0" -Seed 2026
```

8. After the first seed completes and outputs look technically sound, run the publication seed set:

```powershell
.\run_multiseed.ps1 -DataRoot "D:\YOUR\PATH\release_v0"
```

## Do not do these

- do not create a new random train/test split unless it is a separately declared sensitivity experiment;
- do not tune thresholds or preprocessing after seeing the test results;
- do not collapse the seven concepts to binary labels before the primary concept-model experiment;
- do not put Derm7pt images in a public GitHub repository;
- do not interpret the research `predict/query/abstain` output as clinical advice.

Read `README.md`, `docs/EXPERIMENT_PROTOCOL.md`, and `docs/WINDOWS_SETUP.md` before the final publication runs.
