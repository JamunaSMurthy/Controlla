"""Build benchmark-facing metadata artifacts on top of packaged exports."""

from __future__ import annotations

import argparse

from dataset_builder.outputs.benchmark_bundle import write_benchmark_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final benchmark bundle metadata artifacts")
    parser.add_argument("--split-dir", default="outputs/strict_splits")
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--export-summary", default=None)
    parser.add_argument("--stats-report", default=None)
    parser.add_argument("--eval-summary", default=None)
    args = parser.parse_args()
    write_benchmark_bundle(
        args.package_root,
        args.export_summary,
        args.stats_report,
        args.eval_summary,
        split_dir=args.split_dir,
    )


if __name__ == "__main__":
    main()