#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:?Usage: ./run_multiseed.sh /path/to/release_v0}
for SEED in 2026 2027 2028 2029 2030; do
  echo "=== MetaDerm-CBM seed ${SEED} ==="
  ./run_pipeline.sh "$DATA_ROOT" "$SEED"
done
