"""Run the main paper evaluation suite for Controlla.

This is a compatibility wrapper around run_paper_experiments.py.
Prefer:

    python -m controlla.experiments.runners.run_paper_experiments \
        --config experiments/configs/eval_paper.yaml
"""

from __future__ import annotations

from .run_paper_experiments import main


if __name__ == "__main__":
    main()