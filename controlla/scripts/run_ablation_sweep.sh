#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/../.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" -m pipelines.train_controlla --config configs/ablation_full.yaml "$@"
"$PYTHON_BIN" -m pipelines.train_controlla --config configs/ablation_no_identity.yaml "$@"
"$PYTHON_BIN" -m pipelines.train_controlla --config configs/ablation_no_ot.yaml "$@"