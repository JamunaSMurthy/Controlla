# 🎨 Controlla Package

Controlla is the core Python package for multimodal affect-guided human image generation with reference-identity preservation.

[![Dataset](https://img.shields.io/badge/Dataset-AffectHuman--43K-ff6b6b?style=flat-square)](https://huggingface.co/datasets/iamjamuna/AffectHuman-43K)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20coming%20soon-00b894?style=flat-square)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

## ✨ Core Idea

The package combines a diffusion backbone with modular identity, emotion, text, audio, image, graph-fusion, and alignment components. The default configs use a lightweight mock diffusion backbone for fast tests, while the diffusers-backed path is available for full research runs when model weights are available locally.

## 📦 Dataset

Use the public **AffectHuman-43K** dataset:

🔗 [https://huggingface.co/datasets/iamjamuna/AffectHuman-43K](https://huggingface.co/datasets/iamjamuna/AffectHuman-43K)

The benchmark provides 42,469 usable multimodal samples with reference-image identity anchors, affective text/audio controls, emotion labels, and leakage-safe train/validation/test splits.

## 🗂️ Package Layout

- `configs/`: training, inference, and ablation presets
- `data/`: dataset classes and transforms
- `models/`: encoders, graph fusion, OT/alignment hooks, adapters, and diffusion backbone
- `losses/`: identity, emotion, graph, contrastive, and aggregate losses
- `trainers/`: training and evaluation loops
- `pipelines/`: CLI entrypoints for training and inference
- `experiments/`: evaluation runners, metrics, reports, and baseline wrappers
- `utils/`: config, manifest loading, logging, checkpoints, and reproducibility helpers
- `tests/`: smoke and regression tests

## 🚀 Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest tests -q
```

Run a dummy train pass:

```bash
python -m pipelines.train_controlla \
  --config configs/train.yaml \
  --dummy-data
```

Run dummy inference:

```bash
python -m pipelines.infer_controlla \
  --config configs/infer.yaml \
  --prompt "calm portrait"
```

## 👥 Authors

- **Jamuna S. Murthy**, Ramaiah Institute of Technology
- **Amin Karimi Monsefi**, The Ohio State University
- **Rajiv Ramnath**, The Ohio State University

## 📄 Paper

Paper link : [https://arxiv.org/pdf/2605.16603]([https://arxiv.org/pdf/2605.16603)

## 🧭 Notes

- `ot_alignment.py` currently contains a differentiable placeholder for GW/FGW-style alignment.
- External benchmark folders are treated as references and should not be copied into this package release.
- Generated outputs, caches, local datasets, and heavyweight artifacts should stay out of version control.
