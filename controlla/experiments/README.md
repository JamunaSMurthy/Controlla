# Controlla Experiments

This directory contains the reproducible experiments pipeline for Controlla.

## Scope

- Dataset verification, preparation, and unified manifest creation.
- Graph construction for emotion, identity, and cross-modal affinity.
- Metric computation for controllability, identity preservation, cross-modal coherence, disentanglement, and overhead.
- Seeded training and evaluation runners.
- CSV and LaTeX-ready reporting.

## Design Notes

- Raw datasets are expected under `../Datasets` by default.
- Prepared manifests and caches are stored inside `experiments/outputs`.
- CLIP and ImageBind are wrapped through Controlla-side adapters. If checkpoints are unavailable locally, deterministic fallbacks keep the pipeline runnable while making the missing dependency explicit in logs.
- External baselines are exposed through light wrappers first; full heavy reproductions remain marked as TODO where checkpoints or external training are required.

## Recommended Order

1. Run dataset verification and preparation.
2. Build the unified multimodal manifest.
3. Train or register method outputs for seeds `0, 1, 2`.
4. Run main evaluation, ablations, and sensitivity studies.
5. Generate tables and plots.

## Result Tables

The result sections are wired through
`runners/run_paper_experiments.py` and `configs/eval_paper.yaml`. The runner
creates CSV and LaTeX files for:

- main AffectHuman validation/test results
- cross-dataset generalization
- architecture-level comparison
- geometry-aware evaluation
- cross-modal retrieval
- component and modality ablations
- traversal and graph-sensitivity analysis

Run a quick capped smoke check:

```bash
./controlla/experiments/scripts/run_paper_eval.sh controlla/experiments/configs/eval_paper.yaml --max-samples 128
```

Run the full configured evaluation:

```bash
./controlla/experiments/scripts/run_paper_eval.sh
```

Prediction manifests are discovered under:

```text
experiments/outputs/predictions/{section}/{method}/{split}/seed_{seed}/predictions.csv
experiments/outputs/predictions/{method}/{split}/seed_{seed}/predictions.csv
```

For Controlla rows, the AffectHuman manifest can be used as a runnable fallback.
External baselines are listed in the output with `missing_prediction_manifest`
until their generated prediction manifests are added. Missing rows are also
summarized in `experiments/outputs/paper/missing_predictions.csv`.
