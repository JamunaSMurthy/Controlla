# Controlla Dataset Builder

This package builds the pseudo-aligned multimodal dataset used by the Controlla paper.

Current status:

- Step 1 complete: project structure created under `controlla_dataset_builder/`.
- Step 2 complete: schema, constants, config loading, and core utility modules implemented.
- Step 3 complete: raw dataset verification and normalized index builders implemented for the local datasets currently present in this workspace.
- Step 4 complete: text-source generation and post-index label harmonization implemented on top of the normalized indexes.
- Step 5 complete: deterministic audio and identity feature extraction implemented on top of the normalized indexes and generated text manifests.
- Step 6 complete: multimodal alignment implemented using image identity features, audio feature caches, and conditioning text manifests.
- Step 7 complete: deterministic split rebuilding, packaged export manifests, dataset stats, and eval-bench packaging implemented on top of the aligned dataset.
- Step 8 complete: benchmark-facing release metadata implemented, including a machine-readable manifest, artifact checksums, and a benchmark card on top of the packaged exports.

## Goals

- Normalize raw image, audio, and text sources into a shared schema.
- Unify source emotion labels into the Controlla 8-class taxonomy.
- Support identity-aware pseudo alignment across modalities.
- Export reproducible JSONL/CSV manifests for training and evaluation.

## Source Datasets

The default Controlla dataset build is created from the enabled sources in `configs/datasets.yaml`:

- Image sources: `AffectNet`, `RAF-DB`, and `CELEB_HQ`.
- Audio source: `IEMOCAP`.
- Text sources: `IEMOCAP` transcripts and `EmoBank` text annotations.

These sources are harmonized into the shared 8-class emotion taxonomy and then pseudo-aligned into multimodal tuples using cached identity features, audio features, and text manifests.

Additional preprocessors exist for `FFHQ`, `CREMA-D`, `RAVDESS`, `Voxceleb-1`, and `CelebA-Dialog`, but they are disabled in the default configuration.

## Current Commands

Validate the foundational package:

```bash
cd controlla_dataset_builder
python -m pytest tests
```

Build normalized raw indexes:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_raw_indexes
```

Build harmonized text manifests:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_text_sources
```

Build audio and identity feature caches:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_features
```

### Optional real pretrained backends

The builder runs offline by default with deterministic fallback features so tests
and metadata plumbing are reproducible without heavyweight model downloads. To
match the full benchmark-construction protocol with local pretrained encoders,
enable the real backends explicitly:

```bash
pip install -e ".[real-backends]"

export CONTROLLA_USE_REAL_ARCFACE=1
export CONTROLLA_ARCFACE_MODEL_ROOT=/path/to/local/insightface/models

export CONTROLLA_USE_REAL_CLIP=1
export CONTROLLA_CLIP_MODEL=openai/clip-vit-base-patch32

# Optional, when an ImageBind package/checkpoint is installed locally.
export CONTROLLA_USE_REAL_IMAGEBIND=1
```

Alignment metadata records the active backend in `clip_backend` and
`imagebind_backend`, so fallback runs are not silently reported as real
CLIP/ImageBind scoring.

Build aligned multimodal tuples:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_aligned_dataset
```

Package split-aware export manifests:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.export_hf_dataset
```

Build the high-confidence eval bench subset:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_eval_bench
```

Build the final benchmark metadata bundle:

```bash
cd controlla_dataset_builder
python -m dataset_builder.runners.build_benchmark_bundle
```

Inspect configuration loading:

```bash
cd controlla_dataset_builder
python - <<'PY'
from dataset_builder.config import load_yaml_config
cfg = load_yaml_config('configs/build_dataset.yaml')
print(cfg.output_root)
PY
```

Packaged manifests are written under `outputs/aligned/package/`, the eval bench is written under `outputs/aligned/`, and the final benchmark bundle writes `benchmark_manifest.json` plus `BENCHMARK_CARD.md` under `outputs/aligned/package/`.

## License and Responsible Use

The dataset builder is released with the repository under the AffectHuman-43K
research-only license. It is intended for non-commercial academic research,
benchmarking, and evaluation. Do not use the builder or exported dataset for
surveillance, biometric identification deployment, impersonation, identity
misuse, targeted manipulation, or non-consensual synthetic media generation.

Raw dataset files must not be redistributed through the code repository. Keep
generated exports, media, feature caches, and local build artifacts under
ignored `outputs/` or dataset-hosted storage.
