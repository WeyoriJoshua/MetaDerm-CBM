# MetaDerm-CBM v1.1 — Windows execution fix

This maintenance release fixes Windows multiprocessing/DataLoader startup failures caused by executable scripts running at import time under the `spawn` start method.

## Changes
- Added `main()` entry points and `if __name__ == "__main__":` guards to scripts `00`–`13`.
- Added `multiprocessing.freeze_support()` at executable entry points.
- Preserved multi-worker DataLoader support (`num_workers: 4`) on Windows.
- Updated Derm7pt categorical vocabularies in `configs/base.yaml` to match the verified release labels with spaces:
  - `diffuse regular`, `localized regular`, `localized irregular`, `diffuse irregular`
  - `blue areas`, `white areas`
  - `within regression`, `linear irregular`
- Updated reasoning suspicious-state labels to exactly match those vocabularies.
- Re-ran the repository unit tests after patching: 6/6 passed.

## Why this matters
Windows uses a spawn-based multiprocessing model for DataLoader workers. The main module must be safely importable by child processes. Top-level training/evaluation execution caused recursive worker startup and the RuntimeError reported during the first concept-training run.
