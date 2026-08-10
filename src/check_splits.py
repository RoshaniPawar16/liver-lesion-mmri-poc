#!/usr/bin/env python3
"""Build and validate patient-level train/val/test splits.

The JSON contains no official split (verified: only Annotation_info and
Category_info at the top level). We construct a stratified patient-level
split with seed=42, targeting the ~316/78/104 proportions cited in the
original LLD-MMRI paper.

Assertions:
  - Zero patient overlap between any two splits (on normalised IDs)
  - Union of splits == all patients
  - Stratification is on class_id (7 classes) for fine-grained balance

Outputs:
  - manifest.csv updated with split column filled
  - reports/split_report.json with provenance, counts, and per-split tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign and validate splits")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--report", default="reports/split_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    assert "patient_id_norm" in df.columns, "manifest missing patient_id_norm"

    # ── Construct splits ────────────────────────────────────────────────────
    # No official split in JSON (confirmed during investigation).
    # Strategy: stratify on class_id (7 classes) so each fine-grained class
    # is represented in every split.
    # Target: test ~21% (≈104), val ~16% of remainder (≈79), train ~rest (≈315).

    n = len(df)
    y = df["class_id"].values

    # 1. Split off test set (~21%)
    spl_test = StratifiedShuffleSplit(n_splits=1, test_size=0.21, random_state=SEED)
    tv_idx, test_idx = next(spl_test.split(np.zeros(n), y))

    # 2. Split train/val from the remaining set (~20% of remaining ≈ val)
    df_tv = df.iloc[tv_idx]
    y_tv = df_tv["class_id"].values
    spl_val = StratifiedShuffleSplit(n_splits=1, test_size=0.202, random_state=SEED)
    train_sub, val_sub = next(spl_val.split(np.zeros(len(df_tv)), y_tv))

    train_idx = tv_idx[train_sub]
    val_idx = tv_idx[val_sub]

    df = df.copy()
    df["split"] = df["split"].astype(str)  # ensure string dtype before assignment
    df.loc[df.index[train_idx], "split"] = "train"
    df.loc[df.index[val_idx], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"

    # ── Hard assertions: zero patient overlap ────────────────────────────────
    ids = {s: set(df[df["split"] == s]["patient_id_norm"]) for s in ("train", "val", "test")}
    assert ids["train"] & ids["val"] == set(), "OVERLAP: train ∩ val"
    assert ids["train"] & ids["test"] == set(), "OVERLAP: train ∩ test"
    assert ids["val"] & ids["test"] == set(), "OVERLAP: val ∩ test"
    assert ids["train"] | ids["val"] | ids["test"] == set(df["patient_id_norm"]), \
        "Patient count mismatch after split"
    print("ASSERTION PASSED: zero patient overlap between all split pairs")

    # ── Print summary ────────────────────────────────────────────────────────
    split_tables: dict = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        ben = (sub["binary_label"] == 0).sum()
        mal = (sub["binary_label"] == 1).sum()
        print(f"  {split:5s}: n={len(sub):3d}  benign={ben}  malignant={mal}")
        tbl = (
            sub.groupby(["class_id", "class_name", "binary_label"])
            .size()
            .reset_index(name="count")
            .to_dict("records")
        )
        split_tables[split] = tbl

    # ── Write report ─────────────────────────────────────────────────────────
    report = {
        "provenance": (
            "constructed: stratified patient-level split, seed=42, "
            "class_id used for stratification (7 classes). "
            "No official split exists in LLD_MMRI_Annotation.json."
        ),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "split_tables": split_tables,
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSplit report → {out_path}")

    # Update manifest
    df.to_csv(args.manifest, index=False)
    print(f"Manifest updated with split column → {args.manifest}")


if __name__ == "__main__":
    main()
