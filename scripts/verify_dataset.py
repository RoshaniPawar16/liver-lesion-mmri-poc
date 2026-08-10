import os

local_dir = "./LLD-MMRI-MedSAM2"
expected_total = 7971  # from the fetch progress bar

count = 0
for root, dirs, files in os.walk(local_dir):
    count += len(files)

print(f"Files found locally: {count}")
print(f"Expected (per snapshot_download progress bar): {expected_total}")
print(f"Missing: {max(0, expected_total - count)}")

if count >= expected_total:
    print("Looks complete.")
else:
    print("Incomplete — rerun scripts/download_dataset.py to fetch the rest.")
    print("snapshot_download resumes automatically and skips files already present.")
