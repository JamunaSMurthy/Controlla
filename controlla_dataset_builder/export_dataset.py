#!/usr/bin/env python3
"""
Quick export script for Controlla dataset folder structure.

Usage:
    python export_dataset.py \\
        --input outputs/aligned/aligned_dataset.jsonl \\
        --output ../Controlla-Dataset \\
        --anonymize \\
        --use-symlinks
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run dataset export."""
    # Get script directory
    script_dir = Path(__file__).parent.resolve()

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "dataset_builder.runners.export_folder_structure",
    ] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {script_dir}")

    # Run in dataset builder root
    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
