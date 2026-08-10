#!/usr/bin/env bash
# Download the LLD-MMRI-MedSAM2 dataset from Hugging Face.
# Data is CC BY-NC 4.0, not redistributable.
# Requires: pip install huggingface_hub
set -euo pipefail

LOCAL_DIR="${1:-LLD-MMRI-MedSAM2}"

echo "Downloading LLD-MMRI-MedSAM2 → ${LOCAL_DIR}/"
echo "Note: large dataset (~20 GB). Rate-limited? Run: huggingface-cli login"

python3 - <<PYEOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="wanglab/LLD-MMRI-MedSAM2",
    repo_type="dataset",
    local_dir="${LOCAL_DIR}",
)
print(f"Dataset saved to: {path}")
PYEOF
