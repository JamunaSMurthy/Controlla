#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-experiments/configs/datasets.yaml}"

"${PYTHON_BIN}" -m controlla.experiments.datasets.download_datasets --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_emobank --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_iemocap --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_rafd --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_celebahq --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_affectnet --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.prepare_cremad --config "${CONFIG_PATH}"
"${PYTHON_BIN}" -m controlla.experiments.datasets.build_multimodal_manifest --config "${CONFIG_PATH}"
