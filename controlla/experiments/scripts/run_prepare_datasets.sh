#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-experiments/configs/datasets.yaml}"

# Public-release switches:
#   SKIP_DOWNLOAD=1       -> do not run download_datasets
#   ALLOW_MISSING=1       -> continue when optional dataset preparation fails
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[Controlla] Preparing datasets"
echo "[Controlla] Config: ${CONFIG_PATH}"
echo "[Controlla] SKIP_DOWNLOAD=${SKIP_DOWNLOAD}"
echo "[Controlla] ALLOW_MISSING=${ALLOW_MISSING}"

run_step() {
  local module_name="$1"

  echo "[Controlla] Running ${module_name}"

  if [[ "${ALLOW_MISSING}" == "1" ]]; then
    "${PYTHON_BIN}" -m "${module_name}" --config "${CONFIG_PATH}" || {
      echo "[Controlla] WARNING: ${module_name} failed or dataset is missing. Continuing."
    }
  else
    "${PYTHON_BIN}" -m "${module_name}" --config "${CONFIG_PATH}"
  fi
}

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  run_step controlla.experiments.datasets.download_datasets
else
  echo "[Controlla] Skipping dataset download"
fi

run_step controlla.experiments.datasets.prepare_emobank
run_step controlla.experiments.datasets.prepare_iemocap
run_step controlla.experiments.datasets.prepare_rafd
run_step controlla.experiments.datasets.prepare_celebahq
run_step controlla.experiments.datasets.prepare_affectnet
run_step controlla.experiments.datasets.prepare_cremad

# This one should normally succeed if at least one enabled dataset is available.
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  "${PYTHON_BIN}" -m controlla.experiments.datasets.build_multimodal_manifest \
    --config "${CONFIG_PATH}" || {
      echo "[Controlla] ERROR: Failed to build multimodal manifest."
      exit 1
    }
else
  "${PYTHON_BIN}" -m controlla.experiments.datasets.build_multimodal_manifest \
    --config "${CONFIG_PATH}"
fi

echo "[Controlla] Dataset preparation finished"