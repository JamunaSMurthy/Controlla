#!/usr/bin/env bash

set -euo pipefail

python -m pipelines.train_controlla --config configs/train.yaml --dummy-data "$@"