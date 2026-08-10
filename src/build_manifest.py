#!/usr/bin/env python3
"""Build data/manifest.csv from the raw LLD-MMRI-MedSAM2 dataset.

One row per patient. Columns: patient_id_raw, patient_id_norm, class_id,
class_name, binary_label, split (empty; filled by check_splits.py), plus one
path column per phase (empty string when that phase file is absent).

Hard assertions verified in this script:
- num_targets == 1 for every patient-phase entry in the annotation JSON
- JSON Category_info.Malignant matches expected set {1, 3, 6}
- ID normalisation is collision-free (no two raw IDs produce the same normalised key)
- Filename middle digit == JSON category code for every patient
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────

ALL_PHASES: list[str] = [
    "C-pre", "C+A", "C+V", "C+Delay", "DWI", "InPhase", "OutPhase", "T2WI"
]

CATEGORY_TO_NAME: dict[int, str] = {
    0: "Hepatic_hemangioma",
    1: "Intrahepatic_cholangiocarcinoma",
    2: "Hepatic_abscess",
    3: "Hepatic_metastasis",
    4: "Hepatic_cyst",
    5: "FOCAL_NODULAR_HYPERPLASIA",
    6: "Hepatocellular_carcinoma",
}

# Cross-check constant only. Source of truth is Category_info in the JSON.
# If the JSON ever changes, this assertion fires and forces a conscious update.
_EXPECTED_MALIGNANT: frozenset[int] = frozenset({1, 3, 6})


# ── helpers ────────────────────────────────────────────────────────────────

def norm_patient_id(raw: str) -> str:
    """Normalise patient ID: strip hyphens and spaces, uppercase.

    Defensive guard against minor formatting inconsistencies in raw IDs.
    Hyphenated and unhyphenated forms are distinct patients; no merging is
    intended. The corpus IDs normalise to 498 distinct keys with zero
    collisions, verified by assertion in main().
    """
    return raw.replace("-", "").replace(" ", "").upper()


def parse_image_filename(fn: str) -> tuple[str, int, str] | None:
    """Return (patient_id_raw, middle_digit, phase) or None."""
    m = re.match(r"^(MR[- ]?\d+)_(\d+)_(.+)_0000\.nii\.gz$", fn)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def load_json_categories(
    annotation_data: dict,
) -> tuple[dict[str, int], frozenset[int], frozenset[int]]:
    """Extract per-patient category and class sets from parsed annotation data.

    Returns (norm_id -> category_int, malignant_cats, benign_cats).

    Malignant/benign membership is read from Category_info in the JSON
    (source of truth). A cross-check assertion verifies it matches
    _EXPECTED_MALIGNANT.

    Hard assertions:
    - num_targets == 1 for every patient-phase entry.
    - Each patient has a single consistent category across all phases.
    - JSON-derived malignant set matches _EXPECTED_MALIGNANT.
    - Malignant and benign sets are disjoint.
    """
    ai = annotation_data["Annotation_info"]
    ci = annotation_data["Category_info"]

    # Source of truth: read from JSON, not from hand-coded constant
    malignant_cats: frozenset[int] = frozenset(int(x) for x in ci["Malignant"])
    benign_cats: frozenset[int] = frozenset(int(x) for x in ci["Benign"])

    # Cross-check against expected constant
    assert malignant_cats == _EXPECTED_MALIGNANT, (
        f"JSON Category_info.Malignant {sorted(malignant_cats)} "
        f"!= expected {sorted(_EXPECTED_MALIGNANT)}"
    )
    assert malignant_cats & benign_cats == set(), (
        f"Category appears in both Malignant and Benign: {malignant_cats & benign_cats}"
    )

    # num_targets assertion: every patient-phase entry must have exactly one target
    nt_violations: list[tuple[str, int]] = []
    total_entries = 0
    for pid, entries in ai.items():
        for e in entries:
            total_entries += 1
            nt = e["annotation"]["num_targets"]
            if nt != 1:
                nt_violations.append((pid, nt))
    assert len(nt_violations) == 0, (
        f"num_targets != 1 for {len(nt_violations)} patient-phase entries: "
        f"{nt_violations[:5]}"
    )
    print(
        f"ASSERTION PASSED: num_targets == 1 for all {total_entries} "
        f"patient-phase entries"
    )

    out: dict[str, int] = {}
    for pid, entries in ai.items():
        cats: set[int] = set()
        for e in entries:
            for lesion_info in e["annotation"]["lesion"].values():
                cats.add(int(lesion_info["category"]))
        if len(cats) != 1:
            raise AssertionError(
                f"Patient {pid} has inconsistent categories across phases: {cats}"
            )
        out[norm_patient_id(pid)] = next(iter(cats))

    return out, malignant_cats, benign_cats


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/manifest.csv")
    parser.add_argument("--data-dir", default="LLD-MMRI-MedSAM2",
                        help="Root directory of the dataset")
    parser.add_argument("--output", default="data/manifest.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    annotation_path = data_dir / "LLD_MMRI_Annotation.json"

    if not img_dir.exists():
        print(f"ERROR: images dir not found: {img_dir}", file=sys.stderr)
        sys.exit(1)
    if not annotation_path.exists():
        print(f"ERROR: annotation JSON not found: {annotation_path}", file=sys.stderr)
        sys.exit(1)

    # Load annotation JSON once; pass parsed data to helpers
    with open(annotation_path) as f:
        annotation_data = json.load(f)

    # Load categories from JSON; all hard assertions (num_targets, malignant
    # set cross-check) run inside this function
    json_cats, malignant_cats, benign_cats = load_json_categories(annotation_data)

    # Scan image directory: build per-patient phase->path map.
    # Collect unique raw patient IDs to assert collision-free normalisation.
    patient_info: dict[str, dict] = {}
    raw_pid_set: set[str] = set()
    for fn in sorted(img_dir.iterdir()):
        parsed = parse_image_filename(fn.name)
        if parsed is None:
            print(f"WARNING: skipping unexpected filename: {fn.name}", file=sys.stderr)
            continue
        pid_raw, middle, phase = parsed
        raw_pid_set.add(pid_raw)
        norm_id = norm_patient_id(pid_raw)
        if norm_id not in patient_info:
            patient_info[norm_id] = {"pid_raw": pid_raw, "middle": middle, "phases": {}}
        patient_info[norm_id]["phases"][phase] = str(fn)

    # Collision-free assertion: every unique raw ID must map to a unique
    # normalised key. If any two raw IDs collapsed into one norm_id,
    # len(patient_info) < len(raw_pid_set) and this fires.
    assert len(patient_info) == len(raw_pid_set), (
        f"ID normalisation collision: {len(raw_pid_set)} unique raw IDs -> "
        f"{len(patient_info)} normalised IDs (expected equal)"
    )
    print(
        f"ASSERTION PASSED: ID normalisation is collision-free "
        f"({len(raw_pid_set)} raw IDs -> {len(patient_info)} normalised IDs)"
    )

    # ── HARD ASSERTION: middle digit == JSON category for ALL patients ──────
    mismatches: list[tuple] = []
    for norm_id, info in patient_info.items():
        json_cat = json_cats.get(norm_id)
        if json_cat is None:
            raise AssertionError(
                f"Patient {norm_id} present in images/ but absent from JSON"
            )
        if info["middle"] != json_cat:
            mismatches.append((norm_id, info["middle"], json_cat))

    assert len(mismatches) == 0, (
        f"Filename middle digit != JSON category for {len(mismatches)} patient(s): "
        f"{mismatches[:5]}"
    )
    print(
        f"ASSERTION PASSED: filename middle digit == JSON category "
        f"for all {len(patient_info)} patients"
    )

    # ── Build manifest rows ─────────────────────────────────────────────────
    rows: list[dict] = []
    for norm_id, info in sorted(patient_info.items()):
        class_id = info["middle"]
        row: dict = {
            "patient_id_raw": info["pid_raw"],
            "patient_id_norm": norm_id,
            "class_id": class_id,
            "class_name": CATEGORY_TO_NAME[class_id],
            "binary_label": int(class_id in malignant_cats),
            "split": "",
        }
        for phase in ALL_PHASES:
            row[f"path_{phase}"] = info["phases"].get(phase, "")
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\nTotal patients: {len(df)}")
    print("\nClass distribution:")
    for cid in sorted(df["class_id"].unique()):
        n = (df["class_id"] == cid).sum()
        name = CATEGORY_TO_NAME[cid]
        tag = "malignant" if cid in malignant_cats else "benign"
        print(f"  {cid}  {name:<40s}  ({tag})  n={n}")
    n_ben = (df["binary_label"] == 0).sum()
    n_mal = (df["binary_label"] == 1).sum()
    print(f"\nBinary: benign={n_ben}, malignant={n_mal}")

    for phase in ALL_PHASES:
        n_missing = (df[f"path_{phase}"] == "").sum()
        if n_missing:
            print(f"  WARNING: {n_missing} patients missing phase {phase}")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nManifest written -> {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
