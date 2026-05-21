#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-experiments/configs/datasets.yaml}"
MANIFEST_PATH="${2:-}"
OUTPUT_DIR="${3:-experiments/outputs/graphs}"
K_EMOTION="${K_EMOTION:-10}"
K_IDENTITY="${K_IDENTITY:-10}"
K_CROSS_MODAL="${K_CROSS_MODAL:-10}"
DEVICE="${DEVICE:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[Controlla] Building graph priors"
echo "[Controlla] Config: ${CONFIG_PATH}"
echo "[Controlla] Output dir: ${OUTPUT_DIR}"

COMMON_ARGS=(--config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}")

if [[ -n "${MANIFEST_PATH}" ]]; then
  COMMON_ARGS+=(--manifest-path "${MANIFEST_PATH}")
fi

"${PYTHON_BIN}" -m controlla.experiments.graphs.build_emotion_graph \
  "${COMMON_ARGS[@]}" \
  --k "${K_EMOTION}" \
  --node-level sample

"${PYTHON_BIN}" -m controlla.experiments.graphs.build_identity_graph \
  "${COMMON_ARGS[@]}" \
  --k "${K_IDENTITY}" \
  --device "${DEVICE}"

"${PYTHON_BIN}" -m controlla.experiments.graphs.build_cross_modal_graph \
  "${COMMON_ARGS[@]}" \
  --k "${K_CROSS_MODAL}" \
  --device "${DEVICE}"

echo "[Controlla] Graph construction finished"