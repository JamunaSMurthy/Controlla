#!/usr/bin/env python3
"""Build a repeated-reference-identity subset for AffectHuman evaluation.

This companion subset groups samples by the same reference image anchor. It is
intended for evaluating identity consistency across multiple affective controls:
the same visual identity reference appears with 2-5 samples, and each reference
identity group is assigned wholly to one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repeated reference-identity subset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/train_optimized_splits"),
        help="Directory containing train/val/test JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/repeated_identity_subset"),
        help="Output directory for subset JSONL files and stats",
    )
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Include one-sample reference groups to preserve full dataset scale",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def score(sample: dict[str, Any]) -> float:
    return float(sample.get("alignment_scores", {}).get("final_score", 0.0) or 0.0)


def select_diverse_group(samples: list[dict[str, Any]], max_size: int) -> list[dict[str, Any]]:
    """Select up to max_size samples, preferring distinct emotion labels."""
    if max_size <= 0:
        return sorted(samples, key=lambda item: item.get("sample_id", ""))

    by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_emotion[str(sample.get("unified_emotion", "unknown"))].append(sample)

    for emotion_samples in by_emotion.values():
        emotion_samples.sort(key=lambda item: (-score(item), item.get("sample_id", "")))

    selected: list[dict[str, Any]] = []
    for emotion in sorted(by_emotion):
        if len(selected) >= max_size:
            break
        selected.append(by_emotion[emotion][0])

    if len(selected) < max_size:
        selected_ids = {id(sample) for sample in selected}
        remaining = sorted(samples, key=lambda item: (-score(item), item.get("sample_id", "")))
        for sample in remaining:
            if len(selected) >= max_size:
                break
            if id(sample) not in selected_ids:
                selected.append(sample)
                selected_ids.add(id(sample))

    selected.sort(key=lambda item: item.get("sample_id", ""))
    return selected


def assign_group_splits(groups: list[list[dict[str, Any]]], ratios: dict[str, float]) -> dict[str, str]:
    """Assign each reference group to one split with approximate sample ratios."""
    total = sum(len(group) for group in groups)
    targets = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            stable_digest(str(group[0]["reference_image_path"])),
            -len(group),
        ),
    )

    for group in ordered_groups:
        group_key = str(group[0]["reference_image_path"])
        split = max(SPLITS, key=lambda candidate: targets[candidate] - counts[candidate])
        assignments[group_key] = split
        counts[split] += len(group)

    return assignments


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows.extend(read_jsonl(args.input_dir / f"{split}.jsonl"))

    by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reference_path = row.get("reference_image_path")
        if reference_path:
            by_reference[str(reference_path)].append(row)

    selected_groups: list[list[dict[str, Any]]] = []
    min_group_size = 1 if args.include_singletons else args.min_group_size

    for reference_path, samples in by_reference.items():
        if len(samples) < min_group_size:
            continue
        selected = select_diverse_group(samples, max_size=args.max_group_size)
        if len(selected) >= min_group_size:
            reference_id = f"ref_{stable_digest(reference_path)}"
            for sample in selected:
                sample["reference_identity_id"] = reference_id
                sample["original_identity_id"] = sample.get("identity_id")
                sample["identity_id"] = reference_id
            selected_groups.append(selected)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    assignments = assign_group_splits(selected_groups, ratios)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    all_rows: list[dict[str, Any]] = []
    for group in selected_groups:
        split = assignments[str(group[0]["reference_image_path"])]
        for sample in group:
            sample = dict(sample)
            sample["split"] = split
            split_rows[split].append(sample)
            all_rows.append(sample)

    for split in SPLITS:
        split_rows[split].sort(key=lambda item: item.get("sample_id", ""))
        write_jsonl(split_rows[split], args.output_dir / f"{split}.jsonl")

    all_rows.sort(key=lambda item: (item.get("split", ""), item.get("sample_id", "")))
    write_jsonl(all_rows, args.output_dir / "all.jsonl")

    group_sizes = Counter()
    multi_emotion_groups = 0
    for group in selected_groups:
        group_sizes[len(group)] += 1
        if len({sample.get("unified_emotion") for sample in group}) > 1:
            multi_emotion_groups += 1

    stats = {
        "description": "Repeated reference-identity subset for identity consistency evaluation.",
        "grouping_key": "reference_image_path",
        "total_samples": len(all_rows),
        "reference_identity_groups": len(selected_groups),
        "repeated_reference_identity_groups": sum(1 for group in selected_groups if len(group) >= 2),
        "singleton_reference_identity_groups": sum(1 for group in selected_groups if len(group) == 1),
        "group_size_distribution": dict(sorted(group_sizes.items())),
        "multi_emotion_groups": multi_emotion_groups,
        "split_counts": {split: len(split_rows[split]) for split in SPLITS},
        "emotion_counts": dict(sorted(Counter(row.get("unified_emotion") for row in all_rows).items())),
        "identity_leakage_by_reference_group": 0,
        "min_group_size": min_group_size,
        "max_group_size": args.max_group_size,
        "include_singletons": args.include_singletons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "subset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    args = parse_args()
    stats = build_subset(args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
