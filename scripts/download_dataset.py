from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="wanglab/LLD-MMRI-MedSAM2",
    repo_type="dataset",
    local_dir="./LLD-MMRI-MedSAM2"
)

print(f"Dataset downloaded to: {local_path}")
