#!/usr/bin/env bash
# End-to-end smoke test: runs the full pipeline on a 10-patient subset.
# Completes in < 5 min on CPU. No GPU required.
#
# Steps:
#   1. Build manifest (all 498 patients, fast: reads filenames + JSON)
#   2. Build splits
#   3. Cache first 10 patients only (--smoke 10)
#   4. Train smoke_test config (2 epochs, 2 batches, batch_size=4)
#   5. Evaluate
#   6. Grad-CAM on the smoke_test run
#
# Usage: bash scripts/smoke_test.sh
set -euo pipefail

echo "=== Smoke test starting ==="

# 1. Manifest
echo "--- Step 1: build_manifest ---"
python src/build_manifest.py --data-dir LLD-MMRI-MedSAM2 --output data/manifest.csv

# 2. Splits
echo "--- Step 2: check_splits ---"
python src/check_splits.py --manifest data/manifest.csv --report reports/split_report.json

# 3. Cache (10 patients only)
echo "--- Step 3: build_cache (smoke: 10 patients) ---"
python src/build_cache.py \
    --manifest data/manifest.csv \
    --data-dir LLD-MMRI-MedSAM2 \
    --cache-dir data_cache \
    --smoke 10

# 4. Train
echo "--- Step 4: train (smoke config) ---"
python src/train.py --config configs/smoke_test.yaml

# 5. Evaluate
echo "--- Step 5: evaluate ---"
python src/evaluate.py --runs-dir outputs/runs --reports-dir reports

# 6. Grad-CAM (skip if no test predictions or empty test split)
if [ -f outputs/runs/smoke_test/predictions.csv ]; then
    n_test=$(python3 -c "import pandas as pd; print(len(pd.read_csv('outputs/runs/smoke_test/predictions.csv')))")
    if [ "${n_test}" -gt 0 ]; then
        echo "--- Step 6: grad-cam ---"
        python src/gradcam.py \
            --run outputs/runs/smoke_test \
            --manifest data/manifest.csv \
            --n-cases 2
    else
        echo "--- Step 6: grad-cam skipped (empty test set in smoke subset) ---"
    fi
fi

echo "=== Smoke test PASSED ==="
echo "    manifest:      data/manifest.csv"
echo "    splits:        reports/split_report.json"
echo "    cache:         data_cache/ (~20 tensors)"
echo "    metrics:       outputs/runs/smoke_test/metrics_history.csv"
echo "    predictions:   outputs/runs/smoke_test/predictions.csv"
echo "    eval summary:  reports/results_summary.csv"
