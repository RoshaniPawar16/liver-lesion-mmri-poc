#!/usr/bin/env bash
# Run one training experiment.
# Usage: bash scripts/run_train.sh configs/ablation_all_8.yaml [--resume]
#
# Requires: pip install -r requirements.txt
#           data/manifest.csv and data_cache/ must already exist.
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [--resume]}"
RESUME="${2:-}"

echo "=== Training: ${CONFIG} ==="
python src/train.py --config "${CONFIG}" ${RESUME}
