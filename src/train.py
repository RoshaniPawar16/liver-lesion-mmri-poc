#!/usr/bin/env python3
"""Train multi-phase liver lesion binary classifier.

Usage:
    python src/train.py --config configs/ablation_all_8.yaml
    python src/train.py --config configs/ablation_all_8.yaml --resume

All hyperparameters live in the YAML config; the resolved config is written
into the run directory for reproducibility.

Outputs (outputs/runs/<name>/):
    config.yaml          - resolved config snapshot
    last_checkpoint.pt   - updated every epoch (for --resume)
    best_checkpoint.pt   - saved whenever val AUROC improves
    metrics_history.csv  - per-epoch train/val loss and AUROC
    predictions.csv      - test-set patient-level predictions from best ckpt
"""
import argparse
import random
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LiverLesionDataset, collate_fn
from model import MultiPhaseClassifier, count_parameters


# ── Utilities ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: list, device: torch.device) -> torch.Tensor:
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    w_neg = n / (2 * max(n_neg, 1))
    w_pos = n / (2 * max(n_pos, 1))
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


def make_loader(
    manifest: pd.DataFrame,
    cache_dir: str,
    phases: list,
    split: str,
    augment: bool,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple:
    ds = LiverLesionDataset(manifest, cache_dir, phases, split, augment_data=augment)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return ds, loader


# ── Training/eval loop ─────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    train: bool,
    max_batches: int | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """One training or validation pass.

    Returns: (avg_loss, auroc, prob_malignant_array, labels_array)
    """
    model.train(train)
    it = islice(loader, max_batches) if max_batches else loader
    loss_sum = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for phases, labels in it:
            phases = [p.to(device, non_blocking=True) for p in phases]
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(phases)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss_sum += loss.item() * labels.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.cpu())

    if not all_logits:
        return float("nan"), float("nan"), np.array([]), np.array([])

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    probs = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    y = labels_cat.numpy()
    avg_loss = loss_sum / max(len(y), 1)
    try:
        auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    return avg_loss, auroc, probs, y


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for phases, labels in loader:
            phases = [p.to(device) for p in phases]
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(phases)
            all_p.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            all_y.append(labels)
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train liver lesion classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_checkpoint.pt in the run dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f)

    set_seed(cfg["seed"])

    run_dir = Path("outputs/runs") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write resolved config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Run: {cfg['name']}  |  Phases: {cfg['phases']}")

    manifest = pd.read_csv(cfg["manifest"])
    phases: list = cfg["phases"]
    n_phases = len(phases)
    batch_size: int = cfg["batch_size"]
    num_workers: int = cfg.get("num_workers", 0)
    max_batches: int | None = cfg.get("max_batches", None)  # smoke-test limiter
    pin = device.type == "cuda"

    train_ds, train_loader = make_loader(
        manifest, cfg["cache_dir"], phases, "train", True, True, batch_size, num_workers, pin
    )
    val_ds, val_loader = make_loader(
        manifest, cfg["cache_dir"], phases, "val", False, False, batch_size, num_workers, pin
    )
    test_ds, test_loader = make_loader(
        manifest, cfg["cache_dir"], phases, "test", False, False, batch_size, num_workers, pin
    )

    model = MultiPhaseClassifier(n_phases=n_phases).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    cw = class_weights(train_ds.labels, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # State
    start_epoch = 0
    best_val_auroc = -1.0
    patience_counter = 0
    history: list[dict] = []
    last_ckpt = run_dir / "last_checkpoint.pt"

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auroc = ckpt["best_val_auroc"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {ckpt['epoch']}  best_val_auroc={best_val_auroc:.4f}")

    for epoch in range(start_epoch, cfg["max_epochs"]):
        t0 = time.time()
        tr_loss, tr_auc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True, max_batches
        )
        val_loss, val_auc, _, _ = run_epoch(
            model, val_loader, criterion, None, scaler, device, False, max_batches
        )
        dt = time.time() - t0

        row = dict(
            epoch=epoch,
            train_loss=tr_loss, train_auroc=tr_auc,
            val_loss=val_loss, val_auroc=val_auc,
            elapsed_s=round(dt, 1),
        )
        history.append(row)
        print(
            f"Ep {epoch:3d}  tr_loss={tr_loss:.4f} auroc={tr_auc:.4f}  "
            f"val_loss={val_loss:.4f} auroc={val_auc:.4f}  {dt:.1f}s"
        )

        assert not (np.isnan(tr_loss) or np.isinf(tr_loss)), \
            f"NaN/Inf training loss at epoch {epoch}: check data and learning rate"

        state = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            best_val_auroc=best_val_auroc,
            patience_counter=patience_counter,
            history=history,
            cfg=cfg,
        )
        torch.save(state, last_ckpt)

        # Best checkpoint and early stopping
        if not np.isnan(val_auc):
            if val_auc > best_val_auroc:
                best_val_auroc = val_auc
                patience_counter = 0
                torch.save(state, run_dir / "best_checkpoint.pt")
                print(f"  → new best val AUROC: {best_val_auroc:.4f}")
            else:
                patience_counter += 1
                # Early stopping only when not in smoke-test mode
                if not max_batches and patience_counter >= cfg["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch} (patience={cfg['early_stopping_patience']})")
                    break

    # Persist metrics
    pd.DataFrame(history).to_csv(run_dir / "metrics_history.csv", index=False)

    # Ensure best checkpoint exists (smoke mode may not have improved val)
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    if not best_ckpt_path.exists():
        torch.save(state, best_ckpt_path)  # type: ignore[possibly-undefined]

    # Generate test predictions from best checkpoint
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    probs, labels_np = predict(model, test_loader, device)

    pred_df = pd.DataFrame({
        "patient_id_norm": test_ds.patient_ids,
        "prob_malignant": probs,
        "label": labels_np.astype(int),
    })
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    print(f"\nPredictions → {run_dir / 'predictions.csv'}")
    print(f"Done.  Best val AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()
