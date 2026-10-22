#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:?Usage: ./run_pipeline.sh /path/to/release_v0 [seed]}
SEED=${2:-2026}
python scripts/00_audit_data.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED"
python scripts/01_train_concepts.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED"
python scripts/02_train_blackbox.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED"
python scripts/03_fit_cbm_baselines.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED" --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${SEED}_best.pt"
python scripts/04_calibrate.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED" --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${SEED}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${SEED}_best.pt"
python scripts/05_evaluate_metaderm.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED" --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${SEED}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${SEED}_best.pt" --calibration "results/efficientnet_b3/calibration/calibration_seed${SEED}.json"
python scripts/06_run_intervention.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED" --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${SEED}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${SEED}_best.pt"
python scripts/07_run_ablation.py --config configs/base.yaml --data-root "$DATA_ROOT" --seed "$SEED" --ablation all --concept-checkpoint "checkpoints/efficientnet_b3/concepts_seed${SEED}_best.pt" --blackbox-checkpoint "checkpoints/efficientnet_b3/blackbox_seed${SEED}_best.pt" --calibration "results/efficientnet_b3/calibration/calibration_seed${SEED}.json"
python scripts/08_bootstrap_statistics.py --predictions results/efficientnet_b3/final/seed${SEED}/test_predictions.csv --seed "$SEED"
python scripts/09_make_publication_tables.py --results-root results/efficientnet_b3
