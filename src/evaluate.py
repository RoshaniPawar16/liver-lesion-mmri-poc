#!/usr/bin/env python3
"""Evaluate all completed training runs and write summary reports.

For each run with a predictions.csv:
  - AUROC and AUPRC with 1000-resample patient-level bootstrap 95% CIs
  - Binary confusion matrix at Youden-J optimal threshold
  - Calibration curve + ECE for the best-AUROC run
  - reports/results_summary.csv  (ground truth for README numbers)
  - reports/results.md           (formatted table)
  - reports/figures/             (calibration PNG)

Re-running from stored predictions reproduces all numbers byte-identically
(no stochasticity; bootstrap uses a seeded RNG).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# ── Metric helpers ─────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Patient-level bootstrap CI. Returns (mean, lo_2.5, hi_97.5)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    vals: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(float(metric_fn(yt, ys)))
    arr = np.array(vals)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return float(np.mean(arr)), lo, hi


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        err += mask.sum() / n * abs(probs[mask].mean() - y_true[mask].mean())
    return float(err)


def evaluate_run(pred_csv: Path) -> dict | None:
    df = pd.read_csv(pred_csv)
    y = df["label"].values
    p = df["prob_malignant"].values

    if len(np.unique(y)) < 2:
        return None

    auroc, auroc_lo, auroc_hi = bootstrap_ci(y, p, roc_auc_score)
    auprc, auprc_lo, auprc_hi = bootstrap_ci(y, p, average_precision_score)

    thresh = youden_threshold(y, p)
    preds = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return dict(
        auroc=auroc, auroc_lo=auroc_lo, auroc_hi=auroc_hi,
        auprc=auprc, auprc_lo=auprc_lo, auprc_hi=auprc_hi,
        youden_thresh=round(thresh, 4),
        sensitivity=round(tp / max(tp + fn, 1), 4),
        specificity=round(tn / max(tn + fp, 1), 4),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        ece=round(ece(y, p), 4),
        n_test=len(df),
    )


# ── Figures ────────────────────────────────────────────────────────────────

def plot_calibration(run_name: str, pred_csv: Path, fig_dir: Path) -> None:
    df = pd.read_csv(pred_csv)
    y = df["label"].values
    p = df["prob_malignant"].values

    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, "s-", label=f"{run_name}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (malignant)")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve: best model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "calibration_best_model.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration figure → {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all runs")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    reports_dir = Path(args.reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for pred_csv in sorted(runs_dir.glob("*/predictions.csv")):
        run_name = pred_csv.parent.name
        metrics = evaluate_run(pred_csv)
        if metrics is None:
            print(f"SKIP {run_name}: single class in test set")
            continue

        # Read config for phase list
        cfg_path = pred_csv.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            metrics["phases"] = "|".join(cfg.get("phases", []))
            metrics["n_phases"] = len(cfg.get("phases", []))
        else:
            metrics["phases"] = "unknown"
            metrics["n_phases"] = -1

        metrics["run"] = run_name
        results.append(metrics)
        print(
            f"{run_name:40s}  AUROC={metrics['auroc']:.3f} "
            f"[{metrics['auroc_lo']:.3f}-{metrics['auroc_hi']:.3f}]"
        )

    if not results:
        print("No completed runs found. Train first.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("auroc", ascending=False).reset_index(drop=True)

    # Column order for the summary CSV
    cols = [
        "run", "n_phases", "phases",
        "auroc", "auroc_lo", "auroc_hi",
        "auprc", "auprc_lo", "auprc_hi",
        "sensitivity", "specificity",
        "youden_thresh", "tp", "fp", "tn", "fn",
        "ece", "n_test",
    ]
    df[[c for c in cols if c in df.columns]].to_csv(
        reports_dir / "results_summary.csv", index=False
    )
    print(f"\nSaved → {reports_dir / 'results_summary.csv'}")

    # Calibration for best run
    best_run = df.iloc[0]["run"]
    plot_calibration(best_run, runs_dir / best_run / "predictions.csv", fig_dir)

    # Markdown table
    with open(reports_dir / "results.md", "w") as f:
        f.write("# Evaluation results\n\n")
        f.write("Bootstrap CIs are patient-level, 1 000 resamples, seed 42.\n\n")
        f.write("## Ablation table\n\n")
        f.write(
            "| Run | n_phases | AUROC (95 % CI) | AUPRC (95 % CI) "
            "| Sens | Spec | ECE |\n"
        )
        f.write("|-----|----------|-----------------|-----------------|"
                "------|------|-----|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['run']} | {row['n_phases']} "
                f"| {row['auroc']:.3f} [{row['auroc_lo']:.3f}-{row['auroc_hi']:.3f}] "
                f"| {row['auprc']:.3f} [{row['auprc_lo']:.3f}-{row['auprc_hi']:.3f}] "
                f"| {row['sensitivity']:.3f} | {row['specificity']:.3f} "
                f"| {row['ece']:.3f} |\n"
            )
        f.write(f"\nBest model: **{best_run}**\n")
    print(f"Results report → {reports_dir / 'results.md'}")


if __name__ == "__main__":
    main()
