"""PyTorch Dataset for multi-phase liver lesion MRI classification.

One sample = one patient. Each phase is loaded as a (1, D, H, W) float32
tensor from the pre-built cache. Missing phases are returned as zero tensors.
"""
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ALL_PHASES: List[str] = [
    "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase", "T2WI"
]

# Expected shape of every cached tensor (D, H, W)
CACHE_SHAPE: tuple = (32, 64, 64)


# ── Augmentation helpers (train only) ────────────────────────────────────────

def _flip(arr: np.ndarray) -> np.ndarray:
    for ax in range(3):
        if random.random() < 0.5:
            arr = np.flip(arr, axis=ax)
    return arr.copy()


def _intensity_jitter(arr: np.ndarray) -> np.ndarray:
    return arr * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)


def _translate(arr: np.ndarray) -> np.ndarray:
    """Random shift +-2 voxels per axis (wraps; acceptable for ROI crops)."""
    shifts = [random.randint(-2, 2) for _ in range(3)]
    return np.roll(arr, shifts, axis=(0, 1, 2))


def augment(arr: np.ndarray) -> np.ndarray:
    arr = _flip(arr)
    arr = _translate(arr)
    arr = _intensity_jitter(arr)
    return arr


# ── Collation ──────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate a batch of (phases_list, label) into (list_of_batched_phases, labels).

    phases_list has one (1,D,H,W) tensor per phase.
    Result: list of n_phases tensors each shaped (B,1,D,H,W).
    """
    phases_transposed = list(zip(*[item[0] for item in batch]))
    phases_batched = [torch.stack(list(p)) for p in phases_transposed]
    labels = torch.stack([item[1] for item in batch])
    return phases_batched, labels


# ── Dataset ────────────────────────────────────────────────────────────────

class LiverLesionDataset(Dataset):
    """Multi-phase liver lesion MRI dataset.

    Loads float16 .npy tensors from data_cache/, one per patient×phase.
    Returns a list of per-phase tensors plus a binary class label.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: str,
        phases: List[str],
        split: str,
        augment_data: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.phases = phases
        self.do_augment = augment_data

        # One row per patient in the requested split
        df = manifest[manifest["split"] == split].copy()
        df = df.drop_duplicates("patient_id_norm").reset_index(drop=True)
        self.patient_ids: List[str] = df["patient_id_norm"].tolist()
        self.labels: List[int] = df["binary_label"].tolist()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        label = self.labels[idx]

        phase_tensors = []
        for phase in self.phases:
            path = self.cache_dir / f"{pid}_{phase}.npy"
            if path.exists():
                arr = np.load(path).astype(np.float32)
            else:
                arr = np.zeros(CACHE_SHAPE, dtype=np.float32)

            if self.do_augment:
                arr = augment(arr)

            phase_tensors.append(
                torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)
            )

        return phase_tensors, torch.tensor(label, dtype=torch.long)
