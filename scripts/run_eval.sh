#!/usr/bin/env bash
# Evaluate all completed runs and write reports/.
# Re-running this script from stored predictions.csv files is idempotent:
# all numbers (AUROC, CIs, ECE) reproduce byte-identically.
#
# Usage: bash scripts/run_eval.sh [--runs-dir outputs/runs] [--reports-dir reports]
set -euo pipefail

RUNS_DIR="${1:-outputs/runs}"
REPORTS_DIR="${2:-reports}"

echo "=== Evaluating runs in ${RUNS_DIR}/ ==="
python src/evaluate.py --runs-dir "${RUNS_DIR}" --reports-dir "${REPORTS_DIR}"
echo "=== Done. Reports in ${REPORTS_DIR}/ ==="
