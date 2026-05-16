#!/usr/bin/env bash

set -euo pipefail

python -m pipelines.infer_controlla --config configs/infer.yaml --prompt "calm portrait" "$@"