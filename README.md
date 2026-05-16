# 🎨 Controlla

**Multimodal affect-guided human generation with identity preservation**

[![Dataset](https://img.shields.io/badge/Dataset-AffectHuman--43K-ff6b6b?style=flat-square)](https://huggingface.co/datasets/iamjamuna/AffectHuman-43K)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20coming%20soon-00b894?style=flat-square)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## ✨ Overview

**Controlla** is a research codebase for controllable human image generation and editing under multimodal affective conditioning. The system combines a diffusion backbone with identity, emotion, audio, text, and graph-fusion modules so a model can preserve a reference identity while following affective control signals.

The companion dataset is **AffectHuman-43K**, available here:

🔗 **Dataset:** [https://huggingface.co/datasets/iamjamuna/AffectHuman-43K](https://huggingface.co/datasets/iamjamuna/AffectHuman-43K)

---

## 🌈 What Controlla Supports

- **Identity preservation** through reference-image conditioning
- **Affective control** from emotion labels, text prompts, and speech/audio features
- **Graph-based multimodal fusion** across identity, image, text, audio, and emotion branches
- **Optimal-transport-style alignment hooks** for structured cross-modal matching
- **Ablation-ready configs** for identity, graph, and alignment studies
- **Lightweight smoke tests** using a mock diffusion backbone
- **Stable Diffusion integration path** for full research runs when weights are available locally

---

## 📦 Dataset: AffectHuman-43K

**AffectHuman-43K** is an emotion-aligned multimodal benchmark for controlled human affect generation and evaluation.

| Item | Value |
|---|---:|
| Usable samples | **42,469** |
| Modalities | Image, reference image, audio, text |
| Emotion classes | 8 |
| Train / validation / test | 29,728 / 6,371 / 6,370 |
| Cross-split reference-identity leakage | 0 |
| Dataset hub | [AffectHuman-43K](https://huggingface.co/datasets/iamjamuna/AffectHuman-43K) |

The intended sample interpretation is:

```text
reference image -> identity condition
text/audio/emotion label -> affective control condition
target image -> supervision or evaluation target
```

---

## 🧠 Architecture

Controlla is organized as composable modules:

```text
reference image ──► identity encoder ┐
target image ─────► image encoder    │
text prompt ──────► text encoder     ├──► graph fusion ──► diffusion adapter ──► generated image
audio features ───► audio encoder    │
emotion label ────► emotion encoder  │
                                  OT/alignment hooks
```

Available figures live under [`controlla/images`](controlla/images):

| Figure | File |
|---|---|
| Framework overview | [`Figure1.png`](controlla/images/Figure1.png) |
| Module details | [`Figure2.png`](controlla/images/Figure2.png) |
| Additional visualizations | [`Figure3.png`](controlla/images/Figure3.png), [`Figure4.png`](controlla/images/Figure4.png), [`Figure5.png`](controlla/images/Figure5.png), [`Figure6.png`](controlla/images/Figure6.png) |

---

## 🚀 Quick Start

```bash
cd controlla
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest tests -q
```

Run a tiny training smoke test:

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

---

## 🗂️ Project Layout

```text
controlla/
├── configs/              # Training, inference, and ablation presets
├── data/                 # Dataset classes and transforms
├── models/               # Encoders, graph fusion, adapters, diffusion backbone
├── losses/               # Identity, emotion, graph, contrastive, and aggregate losses
├── trainers/             # Train/eval loops
├── pipelines/            # CLI entrypoints
├── experiments/          # Evaluation runners, metrics, reports, and baselines
├── evaluation/           # Evaluation helpers and caching
├── utils/                # Config, manifests, logging, seed, checkpoints
├── tests/                # Smoke and regression tests
└── images/               # Architecture and result figures
```

---

## 📊 Evaluation Areas

Controlla includes reusable components for evaluating:

- **Affective controllability**
- **Identity preservation**
- **Cross-modal consistency**
- **Disentanglement**
- **Retrieval alignment**
- **Latency and overhead**
- **Ablations across modules and modalities**

---

## 👥 Authors

| Author | Affiliation |
|---|---|
| **Jamuna S. Murthy** | Ramaiah Institute of Technology |
| **Amin Karimi Monsefi** | The Ohio State University |
| **Rajiv Ramnath** | The Ohio State University |

---

## 📄 Paper

An arXiv link will be added here when available:

🔗 **Paper:** [https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📚 Citation

```bibtex
@misc{controlla2026,
  title        = {Controlla: Multimodal Affect-Guided Human Generation with Identity Preservation},
  author       = {Murthy, Jamuna S. and Monsefi, Amin Karimi and Ramnath, Rajiv},
  year         = {2026},
  note         = {arXiv link to be added}
}
```

---

## ⚖️ License And Responsible Use

The AffectHuman-43K dataset is released under **CC BY 4.0**. See the dataset page and local dataset license for details.

This project is intended for research on affective controllable generation. Do not use it for impersonation, identity targeting, surveillance, harassment, or non-consensual synthetic media generation.

---

<div align="center">

**⭐ Star the repository if Controlla helps your research.**

Built for careful, reproducible multimodal generation research.

</div>
