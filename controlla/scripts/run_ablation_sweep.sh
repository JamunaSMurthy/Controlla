#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

cd "${REPO_ROOT}"

CONFIGS=(
  "configs/ablation_full.yaml"
  "configs/ablation_no_identity.yaml"
  "configs/ablation_no_ot.yaml"
  "configs/ablation_no_graph.yaml"
  "configs/ablation_no_audio.yaml"
  "configs/ablation_text_only.yaml"
  "configs/ablation_image_text.yaml"
  "configs/ablation_text_audio.yaml"
)

echo "[Controlla] Running ablation training sweep"
echo "[Controlla] Python: ${PYTHON_BIN}"

for CONFIG in "${CONFIGS[@]}"; do
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[Controlla] WARNING: missing config ${CONFIG}; skipping."
    continue
  fi

  echo "[Controlla] Training ablation config: ${CONFIG}"

  "${PYTHON_BIN}" -m controlla.pipelines.train_controlla \
    --config "${CONFIG}" \
    "$@"
done

echo "[Controlla] Ablation sweep finished"