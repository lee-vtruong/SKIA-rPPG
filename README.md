# SKIA-rPPG

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-Research-green)

Official implementation of:

**SKIA: Static Kinematic-Informed Attention for Robust Remote Photoplethysmography under Lower-Face Motion Artifacts**

<p align="center">
  <img src="figures/skia_pipeline.png" width="95%">
</p>

<p align="center">
Overview of the proposed SKIA framework.
</p>

---

# Overview

SKIA-rPPG is a lightweight framework for robust remote photoplethysmography (rPPG) estimation under lower-face motion artifacts.

This project investigates how motion originating from the mouth and jaw regions affects physiological signal reconstruction from facial videos. Based on the observation that lower-face motion introduces substantial non-physiological variations, we propose a simple static lower-face suppression strategy and a lightweight multi-stream deep architecture named **SKIA-Net**.

The framework combines:

- Raw facial appearance information
- Static lower-face suppressed representations
- Temporal-difference motion cues

to improve blood volume pulse (BVP) reconstruction and heart-rate estimation robustness.

---

# Main Contributions

- Systematic analysis of lower-face motion artifacts in rPPG.
- Static lower-face suppression strategy for motion-robust physiological sensing.
- Significant improvement of classical CHROM-based rPPG estimation.
- SKIA-Net: a lightweight multi-stream spatiotemporal network.
- Comprehensive ablation studies:
  - Static vs Dynamic Masking
  - Mask Ratio Selection
  - Input Stream Combinations
- Runtime benchmarking and frequency-domain analysis.

---

# Key Results

## Classical CHROM Analysis

| Method | Pearson ↑ | HR MAE ↓ |
|----------|----------|----------|
| CHROM | 0.3507 | 33.50 |
| CHROM + Static Mask | **0.4166** | **23.44** |

Additional Statistics:

- Winrate: 88.1%
- Wilcoxon Signed-Rank Test: p = 2.0 × 10⁻⁷

---

## SKIA-Net Variants

| Method | Pearson ↑ | HR MAE ↓ |
|----------|----------|----------|
| SKIA Raw | 0.6320 | 8.99 |
| SKIA Masked | 0.6246 | 7.55 |
| SKIA Raw + Masked | 0.6844 | 4.97 |
| SKIA Full | **0.7522** | **3.73** |

---

## Qualitative Reconstruction

<p align="center">
  <img src="figures/qualitative_figure.png" width="85%">
</p>

Example waveform reconstruction produced by SKIA-Net.

---

## Frequency-Domain Analysis

<p align="center">
  <img src="figures/fft_spectrum_comparison.png" width="80%">
</p>

Static lower-face suppression reduces spurious spectral peaks and strengthens the dominant cardiac frequency.

---

## Mask Ratio Ablation

| Retained Facial Ratio | Mean Δ Pearson | Winrate |
|----------|----------|----------|
| 0.5 | -0.0210 | 40.5% |
| 0.6 | +0.0592 | 81.0% |
| 0.7 | **+0.0659** | **88.1%** |
| 0.8 | +0.0410 | 85.7% |

The best performance is achieved when retaining approximately 70% of the upper facial region.

---

# Method Overview

The proposed framework consists of three complementary streams:

### Raw Stream

- Original facial video frames

### Masked Stream

- Static lower-face suppression
- Preserves forehead and cheek regions

### Temporal-Difference Stream

- Frame-to-frame temporal residuals
- Captures short-term motion dynamics

These representations are fused and processed by a lightweight spatiotemporal network to reconstruct the target BVP waveform.

---

# Repository Structure

```text
SKIA-rPPG/
├── src/
├── scripts/
├── utils/
├── figures/
├── results/
├── config.json
├── README.md
└── .gitignore
```

---

# Dataset

This project uses the public UBFC-rPPG dataset.

Dataset files are NOT included in this repository.

Expected structure:

```text
data/
└── UBFC-rPPG/
    └── 1/
        ├── subject1/
        │   ├── vid.avi
        │   └── ground_truth.txt
        ├── subject2/
        └── ...
```

---

# Installation

```bash
pip install torch torchvision numpy scipy pandas matplotlib tqdm opencv-python
```

---

# Data Preprocessing

### Extract Frames

```bash
python scripts/precompute_frames_safe.py
```

### Generate Static Masks

```bash
python scripts/precompute_masks_from_frames.py
```

---

# Training

### SKIA Raw

```bash
python src/train_skia.py --mode raw
```

### SKIA Masked

```bash
python src/train_skia.py --mode masked
```

### SKIA Raw + Masked

```bash
python src/train_skia.py --mode raw_masked
```

### SKIA Full

```bash
python src/train_skia.py --mode full
```

---

# Evaluation

### CHROM Evaluation

```bash
python src/evaluate_static_final.py
```

### Heart Rate Evaluation

```bash
python src/evaluate_skia_hr.py \
  --mode full \
  --ckpt checkpoints/skia_full_best.pt
```

### Mask Ratio Ablation

```bash
python src/evaluate_mask_ratios.py
```

### Dynamic Mask Evaluation

```bash
python src/evaluate_dynamic_soft_mask.py
```

### Runtime Benchmark

```bash
python src/benchmark_runtime.py
```

---

# Experimental Setup

### Dataset

- UBFC-rPPG
- 42 Subjects

### Train/Test Split

Training Subjects: 32

Testing Subjects:

- subject43
- subject44
- subject45
- subject46
- subject47
- subject48
- subject49
- subject5
- subject8
- subject9

### Training Configuration

- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: Negative Pearson Correlation Loss
- Batch Size: 4
- Clip Length: 128
- Input Resolution: 128×128

### Hardware

- NVIDIA A100-SXM4-80GB

---

# Notes

- Dataset files are excluded from version control.
- Frame caches are excluded from version control.
- Model checkpoints are excluded from version control.
- FPS measurements are obtained using GPU inference on preloaded clips.
- Preprocessing and data loading time are excluded.

---

# Future Directions

This work serves as a foundation for future research on:

- Motion-Robust Physiological Sensing
- Speaking-Intensive Physiological Analysis
- Human-Computer Interaction
- Cognitive Load Estimation
- Oral Assessment Systems
- Multimodal Communication Analysis

---

# Citation

```bibtex
@article{le2026skia,
  title={SKIA: Static Kinematic-Informed Attention for Robust Remote Photoplethysmography under Lower-Face Motion Artifacts},
  author={Le, Van-Truong and Huynh, Viet-Tham and Tran, Minh-Triet},
  year={2026}
}
```

(Citation information will be updated after publication.)

---

# License

This repository is released for research and educational purposes.

Dataset usage must comply with the original UBFC-rPPG license and terms of use.
