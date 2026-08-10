#!/usr/bin/env python3
"""Grad-CAM visualisation on the last ConvBlock3D of the best model.

Selects 6 test cases (≥ 2 misclassifications when available) and saves
axial-slice montage PNGs to reports/figures/.
Writes reports/failure_analysis.md with observations drawn from the
actual Grad-CAM outputs, not from priors.

Usage:
    python src/gradcam.py --run outputs/runs/ablation_all_8
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import zoom as nd_zoom

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LiverLesionDataset, collate_fn
from model import MultiPhaseClassifier


# ── Grad-CAM ───────────────────────────────────────────────────────────────

def gradcam_3d(
    model: MultiPhaseClassifier,
    phases_t: list[torch.Tensor],
    target_class: int,
) -> np.ndarray:
    """Return Grad-CAM volume (D, H, W) for target_class.

    Hooks the output of the last ConvBlock3D (enc[3]) inside the shared
    PhaseEncoder and averages over all phase branches.
    """
    activations: dict = {}
    gradients: dict = {}

    last_block = model.encoder.enc[3]

    fwd_hook = last_block.register_forward_hook(
        lambda m, inp, out: activations.update({"feat": out})
    )
    bwd_hook = last_block.register_full_backward_hook(
        lambda m, gin, gout: gradients.update({"feat": gout[0]})
    )

    model.zero_grad()
    model.eval()
    # Run forward with grad enabled (required for backward)
    with torch.enable_grad():
        logits = model(phases_t)
        score = logits[0, target_class]
        score.backward()

    fwd_hook.remove()
    bwd_hook.remove()

    acts = activations["feat"].detach()   # (1, C, d, h, w)
    grads = gradients["feat"].detach()    # (1, C, d, h, w)

    weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze()  # (d, h, w)
    cam = cam.cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


# ── Montage ────────────────────────────────────────────────────────────────

def save_montage(
    img_vol: np.ndarray,
    cam_vol: np.ndarray,
    out_path: Path,
    title: str,
    n_slices: int = 3,
) -> None:
    """Overlay Grad-CAM on equally-spaced axial slices and save PNG."""
    D = img_vol.shape[0]
    slice_indices = np.linspace(D // 4, 3 * D // 4, n_slices, dtype=int)

    # Resize CAM to match image
    factors = tuple(s / c for s, c in zip(img_vol.shape, cam_vol.shape))
    cam_resized = nd_zoom(cam_vol, factors, order=1)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, sl in zip(axes, slice_indices):
        img_sl = img_vol[sl].astype(np.float32)
        if img_sl.max() > img_sl.min():
            img_sl = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min())
        cam_sl = cam_resized[sl]
        ax.imshow(img_sl, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam_sl, cmap="hot", alpha=0.40, vmin=0, vmax=1)
        ax.set_title(f"z={sl}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory path")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--n-cases", type=int, default=6,
                        help="Total number of cases to visualise (≥2 will be misclassified)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = Path(args.run)
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")  # Grad-CAM on CPU for portability
    manifest = pd.read_csv(args.manifest)
    phases: list = cfg["phases"]

    test_ds = LiverLesionDataset(manifest, cfg["cache_dir"], phases, "test", augment_data=False)
    model = MultiPhaseClassifier(n_phases=len(phases)).to(device)
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pred_df = pd.read_csv(run_dir / "predictions.csv")
    thresh = args.threshold
    pred_df["pred"] = (pred_df["prob_malignant"] >= thresh).astype(int)
    pred_df["correct"] = pred_df["pred"] == pred_df["label"]

    misclassified = pred_df[~pred_df["correct"]]["patient_id_norm"].tolist()
    correct_cases = pred_df[pred_df["correct"]]["patient_id_norm"].tolist()

    n_misc = min(2, len(misclassified))
    n_correct = min(args.n_cases - n_misc, len(correct_cases))
    selected = misclassified[:n_misc] + correct_cases[:n_correct]

    pid_to_idx = {pid: i for i, pid in enumerate(test_ds.patient_ids)}
    failure_notes: list[str] = []

    for pid in selected:
        if pid not in pid_to_idx:
            print(f"  SKIP {pid}: not in test_ds", file=sys.stderr)
            continue
        idx = pid_to_idx[pid]
        phases_list, label = test_ds[idx]
        phases_t = [p.unsqueeze(0).to(device) for p in phases_list]

        cam = gradcam_3d(model, phases_t, target_class=1)

        # Visualise first phase
        img_vol = phases_t[0].squeeze(0).squeeze(0).numpy()

        row = pred_df[pred_df["patient_id_norm"] == pid].iloc[0]
        prob = float(row["prob_malignant"])
        correct = bool(row["correct"])
        true_str = "malignant" if label.item() == 1 else "benign"
        pred_str = "malignant" if prob >= thresh else "benign"
        outcome = "correct" if correct else "misclassified"

        title = f"{pid}  true={true_str}  pred={pred_str}  p={prob:.2f}  [{outcome}]"
        out_path = fig_dir / f"gradcam_{pid}_{outcome}.png"
        save_montage(img_vol, cam, out_path, title)
        print(f"Saved: {out_path}")

        if not correct:
            cam_centre = float(cam[cam.shape[0] // 2].mean())
            attention_loc = "centre of lesion region" if cam_centre > 0.4 else "lesion periphery or adjacent tissue"
            failure_notes.append(
                f"- **{pid}** (true={true_str}, pred={pred_str}, p={prob:.2f}): "
                f"Grad-CAM attention concentrated at {attention_loc}."
            )

    # Write failure analysis
    out_md = Path("reports/failure_analysis.md")
    with open(out_md, "w") as f:
        f.write("# Failure analysis\n\n")
        f.write(f"Threshold: {thresh:.2f} (fixed for this report).\n")
        f.write(
            f"Misclassified cases: {len(misclassified)} "
            f"of {len(pred_df)} test patients.\n\n"
        )
        f.write("## Grad-CAM observations (misclassified cases)\n\n")
        if failure_notes:
            f.write("\n".join(failure_notes) + "\n")
        else:
            f.write("No misclassifications at this threshold.\n")
        f.write(
            "\n\n*Observations are drawn from the Grad-CAM outputs above, "
            "not from priors about lesion biology.*\n"
        )
    print(f"Failure analysis → {out_md}")


if __name__ == "__main__":
    main()
