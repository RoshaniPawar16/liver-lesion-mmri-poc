#!/usr/bin/env python3
"""Build ROI tensor cache from full-abdomen NIfTI volumes.

Per patient × phase:
  1. Load image + paired segmentation mask (NIfTI, transposed to D×H×W).
  2. Compute tight bounding box of mask foreground.
  3. Expand bbox by 20% isotropic margin (MIN_MARGIN=2 voxels per side).
  4. Crop image to bbox, resample to CROP_SHAPE=(32,64,64) with linear interp.
  5. Z-score normalise per sample (mu=0, sigma=1; skip div if sigma<1e-8).
  6. Save as float16 .npy tensor.
Writes data_cache/cache_index.csv indexing all saved tensors.

Verified context: volumes are full-abdomen (512×512×72 for most phases,
256×256×24 for DWI). Lesion mask covers ~0.1% of voxels.
Expected cache: ~1 GB for the full dataset (3984 tensors × 256 KB each).
"""
import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

CROP_SHAPE: tuple[int, int, int] = (32, 64, 64)
MARGIN_FRAC: float = 0.20
MIN_MARGIN: int = 2  # voxels per side; prevents zero-width crops on tiny lesions
ALL_PHASES: list[str] = [
    "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase", "T2WI"
]


def mask_bbox_with_margin(
    mask: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Return (z0,z1, y0,y1, x0,x1) bounding box with margin.

    Falls back to a centre-quarter crop when mask is empty (should not occur
    in this dataset, but guards against corrupt labels).
    """
    nz = np.argwhere(mask > 0)
    if len(nz) == 0:
        # Empty mask: centre crop fallback
        shape = np.array(mask.shape)
        ctr = shape // 2
        span = np.maximum(shape // 4, 1)
        lo = np.maximum(ctr - span // 2, 0)
        hi = np.minimum(ctr + span // 2, shape)
        return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])

    mins = nz.min(axis=0)
    maxs = nz.max(axis=0) + 1  # exclusive upper bound

    coords: list[int] = []
    for ax in range(3):
        span = int(maxs[ax] - mins[ax])
        margin = max(MIN_MARGIN, int(np.ceil(span * MARGIN_FRAC)))
        lo = max(0, int(mins[ax]) - margin)
        hi = min(mask.shape[ax], int(maxs[ax]) + margin)
        coords.extend([lo, hi])
    return tuple(coords)  # type: ignore[return-value]


def crop_resample(
    img: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
    target: tuple[int, int, int],
) -> np.ndarray:
    """Crop img to bbox then zoom to target shape (bilinear)."""
    z0, z1, y0, y1, x0, x1 = bbox
    crop = img[z0:z1, y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(target, dtype=np.float32)
    factors = tuple(t / s for t, s in zip(target, crop.shape))
    return zoom(crop.astype(np.float32), factors, order=1)


def zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = float(arr.mean()), float(arr.std())
    return (arr - mu) / sigma if sigma > 1e-8 else arr - mu


def process_one(img_path: str, lbl_path: str) -> np.ndarray | None:
    """Load, crop, resample, normalise → float16 CROP_SHAPE array."""
    try:
        img_nib = nib.load(img_path)
        lbl_nib = nib.load(lbl_path)
    except Exception as exc:
        print(f"  LOAD ERROR {img_path}: {exc}", file=sys.stderr)
        return None

    # NIfTI stores (x,y,z); transpose to (z,y,x) = (depth,height,width)
    img = img_nib.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    lbl = lbl_nib.get_fdata(dtype=np.float32).transpose(2, 1, 0)

    bbox = mask_bbox_with_margin(lbl)
    arr = crop_resample(img, bbox, CROP_SHAPE)
    arr = zscore(arr)
    return arr.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ROI tensor cache")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-dir", default="LLD-MMRI-MedSAM2")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument(
        "--smoke", type=int, default=0,
        help="If >0, process only first N patients (smoke test mode)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.smoke > 0:
        df = df.head(args.smoke)
        print(f"SMOKE MODE: processing first {len(df)} patients")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir = Path(args.data_dir) / "labels"

    n_processed, n_missing, n_failed = 0, 0, 0
    index_rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching"):
        pid = row["patient_id_norm"]
        for phase in ALL_PHASES:
            img_path = row.get(f"path_{phase}", "")
            if not img_path or pd.isna(img_path) or img_path == "":
                n_missing += 1
                continue

            lbl_fn = Path(img_path).name.replace("_0000.nii.gz", ".nii.gz")
            lbl_path = str(lbl_dir / lbl_fn)
            if not os.path.exists(lbl_path):
                print(f"  MISSING label: {lbl_path}", file=sys.stderr)
                n_missing += 1
                continue

            arr = process_one(img_path, lbl_path)
            if arr is None:
                n_failed += 1
                continue

            out_path = cache_dir / f"{pid}_{phase}.npy"
            np.save(out_path, arr)
            index_rows.append({
                "patient_id_norm": pid,
                "phase": phase,
                "class_id": row["class_id"],
                "binary_label": row["binary_label"],
                "split": row["split"],
                "cache_path": str(out_path),
            })
            n_processed += 1

    # Write index
    idx_df = pd.DataFrame(index_rows)
    idx_path = cache_dir / "cache_index.csv"
    idx_df.to_csv(idx_path, index=False)

    # Report size
    total_bytes = sum(
        os.path.getsize(r["cache_path"])
        for _, r in idx_df.iterrows()
        if os.path.exists(r["cache_path"])
    )
    total_gb = total_bytes / 1e9
    print(f"\nCached: {n_processed}  missing: {n_missing}  failed: {n_failed}")
    print(f"Total cache size: {total_gb:.3f} GB")
    if total_gb > 10.0:
        sys.exit(
            "ERROR: cache exceeds 10 GB. Reduce CROP_SHAPE or subsample patients."
        )
    print(f"Index → {idx_path}")


if __name__ == "__main__":
    main()
