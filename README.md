# Liver lesion benign/malignant classification from multi-phase MRI

A self-contained reproducible proof-of-concept for binary classification of liver lesions
from multi-phase MRI, with a phase-ablation study as the central scientific deliverable.

---

## Clinical framing

Dynamic contrast-enhanced MRI is the primary non-invasive tool for liver lesion characterisation.
The canonical HCC signature (arterial-phase hyperenhancement followed by venous or delayed washout)
underpins current LIRADS and EASL criteria.
This project asks two related questions: how much classification signal lives in the full
eight-phase protocol, and which phases contribute most, without assuming the answer in advance.

---

## Data provenance

**Dataset:** LLD-MMRI-MedSAM2 (Lou et al., *Neural Networks* 2025; MedSAM2 annotations, Ma et al. arXiv:2504.03600).
Hosted on [Hugging Face](https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2). Licence: CC BY-NC 4.0.
Data are not redistributed here; download via `bash scripts/download_data.sh`.

- 498 patients, 8 MRI phases each, 3 984 NIfTI volumes total.
- 7 lesion classes: hemangioma (n=79), ICC (n=58), abscess (n=54), metastasis (n=51),
  cyst (n=53), FNH (n=46), HCC (n=157).
- Binary: benign = 232 patients, malignant = 266 patients.
- Segmentation masks are MedSAM2-generated (model-assisted, human-in-the-loop) and are used
  **only for ROI cropping**, never as ground-truth to evaluate against.
- Single-centre Chinese cohort (Zhongshan Hospital); findings may not generalise directly.

**Splits:** No official split exists in the annotation JSON (verified).
Stratified patient-level split constructed with seed 42 (target approx. 316 / 78 / 104 train/val/test);
exact counts in `reports/split_report.json`.

---

## Methods

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ROI crop | Mask bbox + 20 % margin, resampled to 32 x 64 x 64, bilinear | Lesion = 0.1 % of full-abdomen volume |
| Normalisation | Per-sample z-score | Avoids intensity range assumptions across phases |
| Model | Shared-weight 4-block 3D CNN (~1.7 M params) | Simple, auditable; encoder applied once per phase |
| Fusion | Late (concatenate phase features before head) | Enables per-phase attribution |
| Loss | Class-weighted cross-entropy | Mild malignant/benign imbalance (266/232) |
| Optimiser | AdamW, lr=1e-3, wd=0.01 | Standard |
| Training | 60 epochs max, early stopping patience 10 (val AUROC), AMP on CUDA | |
| Seed | 42 (Python / NumPy / PyTorch) | Deterministic |

Augmentation (train only): random axis flips, +-2-voxel translation, +-10 % intensity scale/shift.

---

## Results

### Phase ablation: Experiment A (headline)

Bootstrap CIs are 1 000-resample patient-level, seed 42.
Numbers are read directly from `reports/results_summary.csv`.

| Phase set | n phases | AUROC (95 % CI) | AUPRC (95 % CI) | Sens | Spec |
|-----------|----------|-----------------|-----------------|------|------|
| T2WI + DWI | 2 | <!-- RESULT: ablation_t2wi_dwi_auroc --> | <!-- RESULT: ablation_t2wi_dwi_auprc --> | <!-- RESULT: ablation_t2wi_dwi_sens --> | <!-- RESULT: ablation_t2wi_dwi_spec --> |
| C-pre + C+A | 2 | <!-- RESULT: ablation_pre_a_auroc --> | <!-- RESULT: ablation_pre_a_auprc --> | <!-- RESULT: ablation_pre_a_sens --> | <!-- RESULT: ablation_pre_a_spec --> |
| Contrast 4-phase | 4 | <!-- RESULT: ablation_contrast_4phase_auroc --> | <!-- RESULT: ablation_contrast_4phase_auprc --> | <!-- RESULT: ablation_contrast_4phase_sens --> | <!-- RESULT: ablation_contrast_4phase_spec --> |
| **All 8 phases** | **8** | <!-- RESULT: ablation_all_8_auroc --> | <!-- RESULT: ablation_all_8_auprc --> | <!-- RESULT: ablation_all_8_sens --> | <!-- RESULT: ablation_all_8_spec --> |

### Per-phase: Experiment B (secondary)

| Phase | AUROC (95 % CI) |
|-------|-----------------|
| C-pre | <!-- RESULT: per_phase_C_pre_auroc --> |
| C+A (arterial) | <!-- RESULT: per_phase_C_A_auroc --> |
| C+V (venous) | <!-- RESULT: per_phase_C_V_auroc --> |
| C+Delay | <!-- RESULT: per_phase_C_Delay_auroc --> |
| T2WI | <!-- RESULT: per_phase_T2WI_auroc --> |
| DWI | <!-- RESULT: per_phase_DWI_auroc --> |
| InPhase | <!-- RESULT: per_phase_InPhase_auroc --> |
| OutPhase | <!-- RESULT: per_phase_OutPhase_auroc --> |

> **Fill these numbers:** after running experiments on Kaggle/Colab, copy `reports/results_summary.csv`
> into `reports/` and run `python src/fill_readme.py` to replace the `<!-- RESULT: ... -->` placeholders.

### Calibration (best model)

![Calibration curve](reports/figures/calibration_best_model.png)

ECE (best model): <!-- RESULT: best_ece -->

### Grad-CAM and failure cases

Montage PNGs are in `reports/figures/`. Failure observations are in `reports/failure_analysis.md`.

---

## Limitations

- **Single centre** (Zhongshan Hospital, China): ethnic, scanner, and protocol diversity not represented.
- **ROI cropping presupposes detection**: the pipeline starts from MedSAM2-generated masks. In clinical reality, lesions must first be found.
- **Per-phase ROI extraction, no cross-phase voxel registration**: bounding boxes are computed from each phase's own segmentation mask independently. No cross-phase voxel registration is performed; multi-lesion correspondence is by lesion identity (same patient, same annotation) not by voxel alignment.
- **One random seed**: results should be confirmed across multiple seeds before drawing quantitative conclusions.
- **No radiologist review**: model errors are characterised post-hoc from predictions, not from expert re-reads.
- **MedSAM2 masks are model-generated** (human-in-the-loop but not fully manual): mask quality directly affects crop quality.
- **DWI native resolution**: DWI volumes are 256 x 256 x 24 (vs 512 x 512 x 72 for other phases). After ROI crop and resample to 32 x 64 x 64, the DWI crop covers a smaller physical footprint. This affects crop quality relative to other phases and is a recognised limitation.

---

## What this would become with two years and a clinical dataset

- Whole-liver weakly-supervised detection (MIL) to remove the detection-presupposed bottleneck.
- Cross-phase and CT-MRI registration for genuine multi-modal fusion.
- Fusion with report-derived longitudinal features (prior imaging, AFP trajectory).
- Multi-centre validation and prospective calibration.

---

## Reproduce

```bash
# Build and smoke-test (CPU, under 5 min)
docker build -t liver-lesion-poc .
docker run --rm -v $(pwd)/LLD-MMRI-MedSAM2:/workspace/LLD-MMRI-MedSAM2 \
    liver-lesion-poc bash scripts/smoke_test.sh

# Without Docker
pip install -r requirements.txt
bash scripts/smoke_test.sh

# Full training (GPU recommended, see run_experiments.ipynb)
bash scripts/run_train.sh configs/ablation_all_8.yaml
bash scripts/run_eval.sh
```

**Hardware and wall-clock (approximate, single Kaggle T4):**
- Cache build (3 984 volumes): approx. 45 min
- Training per config (60 epochs, batch 32): approx. 40 min
- All 12 configs: approx. 8 h total

**Built over four evenings as a self-directed exercise to address a stated gap in my
clinical-AI experience before the Cambridge RA application.**

---

Development was assisted by Claude Code; all design decisions, data verification, and interpretation are my own.

---

## Citation

```bibtex
@article{LLD-MMRI,
  title={Sdr-former: A siamese dual-resolution transformer for liver lesion classification using 3d multi-phase imaging},
  author={Lou, Meng and Ying, Hanning and Liu, Xiaoqing and Zhou, Hong-Yu and Zhang, Yuqin and Yu, Yizhou},
  journal={Neural Networks}, pages={107228}, year={2025}
}
@article{MedSAM2,
  title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
  author={Ma, Jun and others},
  journal={arXiv preprint arXiv:2504.03600}, year={2025}
}
```
