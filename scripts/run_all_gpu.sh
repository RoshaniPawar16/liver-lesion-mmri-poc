#!/usr/bin/env bash
# Run all 12 ablation/per-phase configs sequentially on GPU.
# Usage: bash scripts/run_all_gpu.sh --cache-dir /path/to/data_cache
#
# OOM note: if ablation_all_8 OOMs on 8 GB VRAM, halve batch_size in
# configs/ablation_all_8.yaml (32 → 16). All other configs use ≤4 phases.
set -uo pipefail

CACHE_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[[ -z "$CACHE_DIR" ]] && { echo "ERROR: --cache-dir is required"; exit 1; }

CONFIGS=(
    configs/ablation_t2wi_dwi.yaml
    configs/ablation_pre_a.yaml
    configs/ablation_contrast_4phase.yaml
    configs/ablation_all_8.yaml
    configs/per_phase_C_pre.yaml
    configs/per_phase_C_A.yaml
    configs/per_phase_C_V.yaml
    configs/per_phase_C_Delay.yaml
    configs/per_phase_T2WI.yaml
    configs/per_phase_DWI.yaml
    configs/per_phase_InPhase.yaml
    configs/per_phase_OutPhase.yaml
)

mkdir -p outputs/logs
declare -A EXIT_CODES
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="outputs/logs/${name}.log"
    echo "=== ${name} ===" | tee -a "$logfile"
    python src/train.py --config "$cfg" --resume --cache-dir "$CACHE_DIR" \
        2>&1 | tee -a "$logfile"
    EXIT_CODES["$name"]=${PIPESTATUS[0]}
    [[ ${EXIT_CODES["$name"]} -ne 0 ]] \
        && echo "FAILED (exit ${EXIT_CODES[$name]})" | tee -a "$logfile" \
        || echo "DONE" | tee -a "$logfile"
done

echo ""
echo "=== Summary ==="
printf "%-38s %s\n" "Config" "Exit"
printf "%-38s %s\n" "------" "----"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    printf "%-38s %s\n" "$name" "${EXIT_CODES[$name]}"
done
