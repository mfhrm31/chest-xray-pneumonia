# chest-xray-pneumonia 

**Pneumonia Classification from Chest X-Rays Using Transfer Learning**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange)](https://github.com/mfhrm31/chest-xray-pneumonia)

Transfer-learning baselines for binary pneumonia detection from chest X-rays, using ResNet50 and DenseNet121 backbones pretrained on ImageNet.

---

##  Overview

Pneumonia is one of the leading causes of death globally, particularly among children under 5 in low-resource settings. Automated chest X-ray screening can support clinicians in regions with limited radiologist access — a direct match to the use case driving my medical AI research.

This project implements clean transfer-learning baselines on the Kermany et al. (2018) chest X-ray dataset, with reproducible training scripts, calibration analysis, and Grad-CAM interpretability.

Companion work to my research on:
- **[Lung nodule prediction with hybrid feature fusion](https://github.com/mfhrm31/lungnet-hybrid)** (Wiley, 2026)
- **[Medical image preprocessing toolkit](https://github.com/mfhrm31/medical-image-utils)**

##  Goals

- Establish strong transfer-learning baselines on a standard public dataset
- Compare ResNet50 vs DenseNet121 backbones
- Evaluate calibration (ECE, reliability diagrams), not just accuracy
- Provide Grad-CAM visualizations for clinical interpretability
- Demonstrate reproducible medical AI pipeline from data → model → analysis

##  Approach

1. **Data**: Kermany et al. (2018) — 5,863 chest X-ray images, 2 classes (Normal, Pneumonia)
2. **Preprocessing**: Resize to 224×224, ImageNet normalization, histogram equalization
3. **Models**: ResNet50 and DenseNet121, ImageNet pretrained, fine-tuned with frozen-then-unfrozen schedule
4. **Training**: Adam optimizer, cosine LR schedule, weighted cross-entropy for class imbalance
5. **Evaluation**: Accuracy, sensitivity, specificity, AUC, F1, MCC + calibration metrics
6. **Interpretability**: Grad-CAM heatmaps on test set predictions

##  Quickstart

### Installation

```bash
git clone https://github.com/mfhrm31/chest-xray-pneumonia.git
cd chest-xray-pneumonia
pip install -r requirements.txt
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/

### Train a model

```bash
python scripts/train.py --config configs/resnet50.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/best_model.pt
```

##  Project Structure
### Download dataset

The Kermany et al. (2018) chest X-ray dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

After downloading, place files in `data/chest_xray/` with the standard structure:

### Train a model

```bash
python scripts/train.py --config configs/resnet50.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/best_model.pt
```

##  Project Structure

##  Dataset

**Kermany et al. (2018)** — Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning

- 5,863 chest X-ray images (anterior-posterior view)
- 2 classes: Normal, Pneumonia (bacterial + viral)
- Pediatric patients (ages 1–5)
- License: CC BY 4.0
- Citation: Kermany, D. S. et al. (2018). *Cell*.

##  Development Status

| Component | Status |
|---|---|
| Dataset loader | ✅ Implemented |
| ResNet50 baseline | 🚧 In development |
| DenseNet121 baseline | 🚧 In development |
| Calibration analysis | 🚧 Planned |
| Grad-CAM visualizations | 🚧 Planned |

Results will be populated as experiments complete.
